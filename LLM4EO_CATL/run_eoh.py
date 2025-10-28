import os
import shutil
import glob
import time
import inspect
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
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
from typing import List, Dict, Callable, Optional, Any
from Problem.FJSP.FJSP_Problem import FJSP_Problem
from Algrithm.NSGA3 import NSGA_FJSP

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
    ##############################
    # 参数初始化
    n_evals = 2  # 每次评估的重复次数 catl为6不需要调整
    obj_list = ["makespan", "max_load"]
    generation_num = 30  # GA迭代次数
    pop_size = 200  # 种群大小 对其eoh20
    num_evaluators = 3  # 验证并行算子评估器的进程数量
    num_samplers = 3  # 并行调LLM采样器的数量
    max_sample_nums = (
        50  # max_sample_nums 大模型生成算子的个数 达到这个个数或者是迭代次数都会停止
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = "20250831_162523"  # 固定时间戳，便于结果对比
    exp_name = "FJSP_Operator_LLM_Eva"  # 实验名称
    exp_dir = "FJSP_4O_" + timestamp
    log_dir = "outputs/" + exp_dir + "/{}"
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
    llm = HttpsApi(
        host="api.deepseek.com",  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
        key="sk-0e582f6022044b1297882c9d0e1808c1",  # your key, e.g., 'sk-abcdefghijklmn'
        model="deepseek-chat",  # your llm, e.g., 'gpt-3.5-turbo'
    )
    # initialize templates
    template_dict = {
        "ma_crossover_template": ma_crossover_template,
        "ma_mutation_template": ma_mutation_template,
        "op_crossover_template": op_crossover_template,
        "op_mutation_template": op_mutation_template,
    }
    # operators = {
    #     "op_crossover": op_crossover,
    #     "op_mutation": op_mutation,
    #     "ma_crossover": ma_crossover,
    #     "ma_mutation": ma_mutation,
    # }
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
        print(f"\n===== Evaluating instance: {instance_name} =====")

        problem = FJSP_Problem(obj_list, instance_path)  # initialize problem\

        for operator_t, template in template_dict.items():
            # filter only ma_crossover and ma_mutation
            if operator_t not in ["op_mutation_template"]:
                continue

            evo_operator = operator_t[:-9]  # remove "_template"
            # credate evaluator
            evaluator = MyEvaluation(
                algorithm=NSGA_FJSP,
                problem=problem,
                exp_name=exp_name,
                template=template,  # use the first template as initial, and it can be changed in the process
                ev_operator_name=evo_operator,  # need to be evolution operator
                ref_point=problem.ref_points,
                generation_num=generation_num,
                n_evals=n_evals,
                pop_size=pop_size,
            )
            # set only one operator
            # evaluator.set_operator(evo_operator, operators[evo_operator])
            # set all operators, and type is callable
            # evaluator.set_operator("op_crossover", op_crossover)
            # evaluator.set_operator("op_mutation", op_mutation)
            # evaluator.set_operator("ma_crossover", ma_crossover)
            # evaluator.set_operator("ma_mutation", ma_mutation)
            evaluator.save_operatorstr_bydict(default_operators_source)

            log_path = str(get_path(log_dir.format(evo_operator)))
            manage_directory(log_path)

            profiler = TensorboardProfiler(
                wandb_project_name="aad-catl",
                log_dir=log_path,
                name=f"CATL@eoh@nsga_{evo_operator}",
                create_random_path=False,
            )

            # eoh evo
            eoh = EoH(
                evaluation=evaluator,
                num_evaluators=num_evaluators,  # 指定线程池的最大工作线程数，即同时执行评估任务的线程数量上限
                num_samplers=num_samplers,
                max_sample_nums=max_sample_nums,
                profiler=profiler,
                # llm=GPT4oMini(),
                # llm=DGX2API(),
                llm=llm,
                debug_mode=True,
            )
            eoh.run()
