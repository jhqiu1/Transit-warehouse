import os
import shutil
import glob
import json
import time
import inspect
from copy import deepcopy
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH
from llm4ad.tools.profiler import WandBProfiler, TensorboardProfiler
from _evaluate import (
    MyEvaluation,
    ma_crossover_template,
    ma_mutation_template,
    op_crossover_template,
    op_mutation_template,
    op_crossover,
    op_mutation,
    ma_crossover,
    ma_mutation,
)

# from _storage import GlobalResultStorage
from typing import List, Dict, Callable, Optional, Any
from Problem.FJSP.FJSP_Problem import FJSP_Problem
from Algrithm.NSGA2 import NSGA_FJSP
from logger.llm4eo_logger import ExperimentLogger

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_path(*relative_parts):
    return os.path.join(PROJECT_ROOT, *relative_parts)


def manage_directory(path):
    """
    检查指定的文件夹是否存在，如果存在则删除，然后重新创建。
    如果文件夹不存在，则直接创建。

    :param path: 文件夹路径
    """
    if os.path.exists(path):
        # 如果文件夹存在，先删除
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")

    # 重新创建文件夹
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")


if __name__ == "__main__":
    # main算法核心逻辑介绍：算子组合搜索，借助坐标轮换法，搜索最优算子组合
    ##############################
    # Eoh paras
    n_evals = 5  # 每次评估的重复次数 catl为6不需要调整
    obj_list = ["makespan", "total_load"]
    generation_num = 30  # GA迭代次数
    pop_size = 100  # 种群大小 对其eoh20
    eoh_max_generations = (
        5  # # max_sample_nums 大模型生成算子的个数 达到这个个数或者是迭代次数都会停止
    )
    eoh_max_sample_nums = 25  # 最小评估次数
    eoh_pop_size = 5  # 最小种群
    eoh_num_samplers = 2  # 单线程
    eoh_num_evaluators = 2  # 单线程
    # set Univariate search technique paras
    max_cycles = 6  # 最大迭代轮数
    convergence_threshold = 0.01  # 收敛阈值，分数提升小于此值则停止
    operators_to_optimize = [
        "op_crossover",
        "op_mutation",
        "ma_crossover",
        "ma_mutation",
    ]  # need to be evolution operator
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = "OC2"  # fixed timestamp for result comparison
    exp_name = "FJSP_OC_TMAX"
    exp_dir = "FJSP_OC_" + timestamp
    log_dir = "outputs/" + exp_dir + "/eoh_log/{}"
    # 定义基准目录
    benchmark = "brandimarte"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(BASE_DIR, "Instances", benchmark)

    # 获取所有算例文件路径
    instance_paths = glob.glob(os.path.join(instances_dir, "*.txt"))
    instance_paths.sort()
    # print instance paths
    print(f"Found {len(instance_paths)} instances in '{benchmark}' benchmark:")
    for i, path in enumerate(instance_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    ##############################
    # initialize LLM
    llm = HttpsApi(None, None, None)
    # llm._model = "deepseek-v3"
    # llm.draw_sample("你是什么模型，什么版本，什么供应厂商的")
    # initialize templates
    template_dict = {
        "ma_crossover_template": ma_crossover_template,
        "ma_mutation_template": ma_mutation_template,
        "op_crossover_template": op_crossover_template,
        "op_mutation_template": op_mutation_template,
    }
    # 获取默认算子的源代码
    default_operators_source = {
        "op_crossover": inspect.getsource(op_crossover),
        "op_mutation": inspect.getsource(op_mutation),
        "ma_crossover": inspect.getsource(ma_crossover),
        "ma_mutation": inspect.getsource(ma_mutation),
    }

    # 评估所有算例
    all_results = {}
    for instance_path in instance_paths[14:15]:  # only test one instance

        instance_name = os.path.basename(instance_path)  # get instance name
        # 创建日志记录器
        logger = ExperimentLogger(exp_dir, instance_name)
        logger.log_info(f"===== Evaluating instance: {instance_name} =====")

        problem = FJSP_Problem(obj_list, instance_path)  # initialize problem
        # initialize evolution operators using code string
        # credate evaluator
        evaluator = MyEvaluation(
            algorithm=NSGA_FJSP,
            problem=problem,
            exp_name=exp_name,
            template=None,  # it can be changed in the evolution process
            ev_operator_name=None,  # it can be changed in the evolution process
            ref_point=problem.ref_points,
            generation_num=generation_num,
            n_evals=n_evals,
            pop_size=pop_size,
        )

        # evaluate initial combination score
        initial_score = evaluator.evaluate_combination(default_operators_source) * 0.9
        # initial_score = 0
        best_score = initial_score
        current_operators = deepcopy(default_operators_source)
        best_operators = deepcopy(current_operators)

        logger.log_info(f"Initial combination score: {initial_score}")

        # 记录初始状态
        logger.record_iteration(0, "initial", current_operators, initial_score)
        logger.record_best_result(0, "initial", best_operators, best_score)

        # 在循环开始前初始化最佳结果列表
        best_results = []
        # 坐标轮换法主循环
        for cycle in range(max_cycles):
            logger.log_info(f"--- Starting coordinate cycle {cycle+1}/{max_cycles} ---")
            score_improved = False
            # 依次优化每个算子
            for operator_name in operators_to_optimize:
                logger.log_info(f"Optimizing {operator_name}...")
                template = template_dict[operator_name + "_template"]

                # 创建针对当前算子的评估器
                operator_evaluator = MyEvaluation(
                    algorithm=NSGA_FJSP,
                    problem=problem,
                    exp_name=exp_name,
                    template=template,
                    # template=default_operators_source[operator_name],
                    ev_operator_name=operator_name,
                    ref_point=problem.ref_points,
                    generation_num=generation_num,
                    n_evals=n_evals,
                    pop_size=pop_size,
                )
                # 更新当前operators set
                operator_evaluator.save_operatorstr_bydict(current_operators)
                waitval_operators = deepcopy(current_operators)

                # logger set
                log_path = str(
                    get_path(log_dir.format(f"cycle{cycle+1}_{operator_name}"))
                )
                manage_directory(log_path)
                profiler = TensorboardProfiler(
                    wandb_project_name="aad-catl",
                    log_dir=log_path,
                    name=f"CATL@eoh@nsga_{operator_name}_cycle{cycle+1}",
                    create_random_path=False,
                )
                # 运行EOH优化当前算子
                eoh = EoH(
                    evaluation=operator_evaluator,
                    profiler=profiler,
                    llm=llm,
                    debug_mode=True,
                    max_generations=eoh_max_generations,
                    max_sample_nums=eoh_max_sample_nums,
                    pop_size=eoh_pop_size,
                    num_samplers=eoh_num_samplers,
                    num_evaluators=eoh_num_evaluators,
                )
                # run eoh for searching this operator
                eoh.run()
                ope_pops = eoh._population

                # 获取最佳函数及其分数
                if ope_pops and len(ope_pops.population) > 0:  # 确保种群不为空
                    best_function = ope_pops.population[0]  # 第一个个体就是分数最高的
                    best_score_val = best_function.score

                    # 检查分数是否有效
                    if best_score_val is not None and best_score_val > float("-inf"):
                        # 更新当前算子为源代码字符串
                        waitval_operators[operator_name] = str(best_function)
                        logger.log_info(
                            f"Best {operator_name} score: {best_score_val:.4f}"
                        )
                    else:
                        logger.log_warning(
                            f"Invalid score for best {operator_name}, keeping current operator"
                        )
                else:
                    logger.log_warning(
                        f"No valid {operator_name} found during EOH optimization, keeping current operator"
                    )

                # evaluate new combination score
                new_score = operator_evaluator.evaluate_combination(
                    waitval_operators
                )  # it will change the default set in operator_evaluator
                logger.log_info(
                    f"After optimizing {operator_name}, score: {new_score:.4f}"
                )

                # 记录当前迭代
                iteration_data = logger.record_iteration(
                    cycle + 1, operator_name, current_operators, new_score
                )

                # update best if improved
                if new_score > best_score:
                    best_score = new_score
                    current_operators = deepcopy(
                        waitval_operators
                    )  # will be update and application in next operator evo process
                    best_operators = deepcopy(current_operators)  # record
                    score_improved = True
                    logger.log_info(f"New best score: {best_score:.4f}")
                    logger.log_info(
                        f"Operator {operator_name}, has been updated and used !"
                    )

                    # 记录最佳结果
                    best_result = logger.record_best_result(
                        cycle + 1, operator_name, best_operators, best_score
                    )

            # check convergence based on score change
            # if cycle > 0:
            #     score_change = abs(new_score - previous_score)
            #     if score_change < convergence_threshold:
            #         logger.log_info(
            #             f"Score change ({score_change}) below threshold ({convergence_threshold}), stopping."
            #         )
            #         break
            previous_score = new_score

        logger.log_info(f"Optimization completed. Best score: {best_score}")
        logger.log_info("Best operator combination:")
        for op_name, op_source in best_operators.items():
            if isinstance(op_source, str):
                logger.log_info(
                    f"  {op_name}: source code length = {len(op_source)} characters"
                )
            else:
                logger.log_info(f"  {op_name}: default operator")

        # save final results of this instance
        all_results[instance_name] = {
            "best_score": best_score,
            "best_operators": {
                k: v if isinstance(v, str) else "default"
                for k, v in best_operators.items()
            },
            "improvement_history": logger.best_results,
            "all_iterations": logger.all_iterations,
        }

    # save all instances results
    logger.save_final_results(all_results)
