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
from LLM.Prompts import Prompts
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
from _storage import GlobalResultStorage
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
    # main算法核心逻辑介绍：每轮决策的时候决定搜索哪个算子i和采样几次（max_sample_nums）
    ##############################
    # Eoh paras
    n_evals = 2  # 每次评估的重复次数 catl为6不需要调整
    obj_list = ["makespan", "max_load"]
    generation_num = 1  # GA迭代次数
    pop_size = 200  # 种群大小 对其eoh20
    num_evaluators = 2  # 验证并行算子评估器的进程数量
    num_samplers = 2  # 并行调LLM采样器的数量
    max_sample_nums = (
        1  # max_sample_nums 大模型生成算子的个数 达到这个个数或者是迭代次数都会停止
    )
    # set Univariate search technique paras
    max_cycles = 50  # 最大迭代轮数
    convergence_threshold = 0.01  # 收敛阈值，分数提升小于此值则停止
    operators_to_optimize = [
        "op_crossover",
        "op_mutation",
        "ma_crossover",
        "ma_mutation",
    ]  # need to be evolution operator
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = "20250906_140943"
    # timestamp = "debug"  # fixed timestamp for result comparison
    exp_name = "FJSP_OC_LLM_Eva"
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
    Prompt = Prompts()
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
        initial_score = evaluator.evaluate_combination(default_operators_source)
        best_score = initial_score
        current_operators = deepcopy(default_operators_source)
        best_operators = deepcopy(current_operators)

        logger.log_info(f"Initial combination score: {initial_score}")

        # 记录初始状态
        logger.record_iteration(0, "initial", current_operators, initial_score)
        logger.record_best_result(0, "initial", best_operators, best_score)

        # 基于 MDP 的思路：每轮先评估每个算子的单坐标改动，轮末只应用最优那个
        for cycle in range(max_cycles):
            logger.log_info(f"--- Starting coordinate cycle {cycle+1}/{max_cycles} ---")

            # 本轮最佳候选（单坐标改动后的一整套 operators）
            best_local_score = float("-inf")
            best_local_operators = None
            best_local_op = None

            # 依次优化每个算子（子轮）
            for operator_name in operators_to_optimize:
                logger.log_info(f"Optimizing {operator_name}...")

                raw = llm.draw_sample(
                    Prompt.prompt_newTemplate(
                        current_operators[operator_name],
                        template_dict[operator_name + "_template"],
                    )
                )
                text = raw.strip()
                if text.startswith("```"):
                    nl = text.find("\n")
                    text = text[nl + 1 :] if nl != -1 else text
                    if text.strip().endswith("```"):
                        text = text[: text.rfind("```")]
                template = text.strip()

                # 针对当前算子构造评估器（只变这个算子）
                operator_evaluator = MyEvaluation(
                    algorithm=NSGA_FJSP,
                    problem=problem,
                    exp_name=exp_name,
                    template=template,
                    ev_operator_name=operator_name,
                    ref_point=problem.ref_points,
                    generation_num=generation_num,
                    n_evals=n_evals,
                    pop_size=pop_size,
                )
                # 把当前整套算子写入评估器作为默认
                operator_evaluator.save_operatorstr_bydict(current_operators)
                waitval_operators = deepcopy(current_operators)

                # logger/profiler
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

                # 运行 EOH 搜索该算子
                eoh = EoH(
                    evaluation=operator_evaluator,
                    num_evaluators=num_evaluators,
                    num_samplers=num_samplers,
                    max_sample_nums=max_sample_nums,
                    profiler=profiler,
                    llm=llm,
                    debug_mode=True,
                )
                eoh.run()

                # —— 获取最优个体（优先用公开方法；没有就兜底 _population）——
                try:
                    ope_pops = getattr(
                        eoh, "get_population", lambda: getattr(eoh, "_population", None)
                    )()
                except Exception:
                    ope_pops = getattr(eoh, "_population", None)

                # 无可行体 → 跳过该算子
                if not ope_pops or len(ope_pops.population) == 0:
                    logger.log_warning(
                        f"No valid {operator_name} found during EOH optimization; skip"
                    )
                    continue

                best_function = ope_pops.population[0]  # 假设已按分数降序
                best_score_val = getattr(best_function, "score", None)
                if best_score_val is None or best_score_val == float("-inf"):
                    logger.log_warning(f"Invalid score for best {operator_name}; skip")
                    continue

                # 构造“只改这个算子”的候选组合
                waitval_operators[operator_name] = str(best_function)
                logger.log_info(
                    f"Best {operator_name} (local) score: {best_score_val:.4f}"
                )

                # 评估这套候选组合（注意：记录的就是候选本身）
                new_score = operator_evaluator.evaluate_combination(waitval_operators)
                logger.log_info(
                    f"After optimizing {operator_name}, score: {new_score:.4f}"
                )
                logger.record_iteration(
                    cycle + 1, operator_name, waitval_operators, new_score
                )

                # 更新本轮最优候选
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_operators = deepcopy(waitval_operators)
                    best_local_op = operator_name

            # —— 主轮收尾：一次性应用或保持不变 ——
            if best_local_operators is not None and best_local_score > best_score:
                best_score = best_local_score
                current_operators = deepcopy(best_local_operators)
                best_operators = deepcopy(best_local_operators)
                logger.log_info(
                    f"[Cycle {cycle+1}] New global best: {best_score:.4f} (by {best_local_op})"
                )
                logger.record_best_result(
                    cycle + 1, best_local_op, best_operators, best_score
                )
            else:
                logger.log_info(
                    f"[Cycle {cycle+1}] No improvement; keep current operators"
                )

            # —— 可选：早停（按“本轮提升”阈值）——
            if best_local_operators is not None:
                improvement = (
                    best_local_score - best_score
                )  # 此时 best_score 已更新或保持不变
                if abs(improvement) < convergence_threshold:
                    logger.log_info(
                        f"Early stop: improvement {improvement:.6f} < threshold {convergence_threshold}"
                    )
                    break

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
