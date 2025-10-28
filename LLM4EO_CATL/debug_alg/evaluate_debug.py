import sysconfig
import os
import json
import copy
import random
import time
import glob
import sys
import dill
from typing import List, Dict, Callable, Optional, Any
import concurrent.futures
from copy import deepcopy

# solving the import issue
sys.path.insert(0, sys.path[0] + "/../")
from Problem.FJSP.FJSP_Problem_code2 import FJSP_Problem
from Algrithm.NSGA3_newcode import NSGA_FJSP
from llm4ad.base import Evaluation
from utils.code_json_to_program import (
    function_string_to_program,
    function_to_callable,
    load_best_operators_from_json,
)
from _evaluate import (
    ma_crossover,
    ma_mutation,
    op_crossover,
    op_mutation,
)
import inspect
import geatpy as ea
import numpy as np


ma_crossover_json = 'def ma_crossover(parent1: np.ndarray, parent2: np.ndarray, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:\n    """\u673a\u5668\u5206\u914d\u4ea4\u53c9\u7b97\u5b50\n    Args:\n        parent1: \u7236\u4ee31\u7684\u673a\u5668\u9009\u62e9\u67d3\u8272\u4f53 (n_vars,)\n        parent2: \u7236\u4ee32\u7684\u673a\u5668\u9009\u62e9\u67d3\u8272\u4f53 (n_vars,)\n        n_vars: \u67d3\u8272\u4f53\u957f\u5ea6\n    Returns:\n        \u4e24\u4e2a\u5b50\u4ee3\u67d3\u8272\u4f53 (child1, child2)\n    """\n    child1 = np.zeros(n_vars, dtype=parent1.dtype)\n    child2 = np.zeros(n_vars, dtype=parent2.dtype)\n    \n    conflict_mask = parent1 != parent2\n    non_conflict_mask = parent1 == parent2\n    \n    child1[non_conflict_mask] = parent1[non_conflict_mask]\n    child2[non_conflict_mask] = parent2[non_conflict_mask]\n    \n    conflict_indices = np.where(conflict_mask)[0]\n    np.random.shuffle(conflict_indices)\n    \n    for i, idx in enumerate(conflict_indices):\n        if i % 2 == 0:\n            child1[idx] = parent1[idx]\n            child2[idx] = parent2[idx]\n        else:\n            child1[idx] = parent2[idx]\n            child2[idx] = parent1[idx]\n    \n    return child1, child2\n\n'
ma_mutation_json = 'def ma_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:\n    """\u673a\u5668\u5206\u914d\u53d8\u5f02\u7b97\u5b50\n    Args:\n        solution: \u673a\u5668\u9009\u62e9\u67d3\u8272\u4f53 (n_vars,)\n        n_vars: \u67d3\u8272\u4f53\u957f\u5ea6\n    Returns:\n        \u53d8\u5f02\u540e\u7684\u67d3\u8272\u4f53\n    """\n    mutated = solution.copy()\n    max_machine = np.max(solution)\n    for i in range(n_vars):\n        mutation_prob = 1.0 / (n_vars * (1 + np.log1p(max_machine - 1)))\n        if np.random.rand() < mutation_prob:\n            valid_machines = list(range(1, max_machine + 1))\n            if solution[i] in valid_machines:\n                valid_machines.remove(solution[i])\n            if valid_machines:\n                mutated[i] = np.random.choice(valid_machines)\n    return mutated\n\n'
op_crossover_json = 'def op_crossover(par ent1: np.ndarray, parent2: np.ndarray, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:\n    """\u5de5\u5e8f\u4f18\u5148\u7ea7\u4ea4\u53c9\uff08\u8fde\u7eed\u5b9e\u6570\u5411\u91cf\uff09\u3002\u5355\u70b9\u5207\u7247\u62fc\u63a5\uff0c\u8fd4\u56de\u4e0e\u7236\u4ee3\u540c\u957f\u7684\u4e24\u4e2a\u5b50\u4ee3\u3002\n    \u7ea6\u675f\uff1a\n      - parent1/parent2 \u90fd\u88ab\u89c6\u4e3a 1D \u8fde\u7eed\u5b9e\u6570\u5411\u91cf\uff0c\u4e0d\u662f\u6392\u5217\uff1b\u4e0d\u505a\u67e5\u91cd\u4e0e\u8865\u9f50\u3002\n      - \u53ea\u505a\u7b80\u5355\u5207\u5206\u62fc\u63a5\uff0c\u5e76\u5bf9\u7ed3\u679c clip \u5230 [0,1] \u4ee5\u589e\u5f3a\u9c81\u68d2\u6027\u3002\n      - \u4efb\u4f55\u5f02\u5e38/\u5c3a\u5bf8\u4e0d\u7b26\uff0c\u76f4\u63a5\u8fd4\u56de\u7236\u4ee3\u62f7\u8d1d\u3002\n    # \u53ea\u4f7f\u7528 numpy\u3002\u628a\u67d3\u8272\u4f53\u5f53\u4f5c\u957f\u5ea6\u4e3a n_vars \u7684\u4e00\u7ef4\u8fde\u7eed\u5b9e\u6570\u5411\u91cf\uff08\u53d6\u503c\u901a\u5e38\u5728 [0,1]\uff09\u3002\n    # \u4e25\u7981\u628a\u5b83\u5f53\u6210\u201c\u65e0\u91cd\u590d\u6392\u5217\u201d\uff0c\u4e25\u7981\u4f7f\u7528\uff1a\u67e5\u91cd(in / set)\u3001\u5360\u4f4d(-1)\u3001\u5faa\u73af\u586b\u5145\u7b49\u903b\u8f91\u3002\n    # \u4efb\u4f55\u5f02\u5e38\u6216\u7ef4\u5ea6\u4e0d\u7b26\u65f6\uff0c\u5fc5\u987b\u539f\u6837\u8fd4\u56de\u7236\u4ee3\uff08\u4fdd\u6301\u957f\u5ea6\u4e0e dtype\uff09\u3002\n\n    """\n    if parent1.shape != (n_vars,) or parent2.shape != (n_vars,):\n        return parent1.copy(), parent2.copy()\n    \n    try:\n        crossover_point = np.random.randint(1, n_vars)\n        \n        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])\n        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])\n        \n        child1 = np.clip(child1, 0, 1)\n        child2 = np.clip(child2, 0, 1)\n        return child1, child2\n    except:\n        return parent1.copy(), parent2.copy()\n\n'
op_mutation_json = 'def op_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:\n    """\u5de5\u5e8f\u4f18\u5148\u7ea7\u53d8\u5f02\u7b97\u5b50\n    Args:\n        solution: \u5de5\u5e8f\u4f18\u5148\u7ea7\u67d3\u8272\u4f53 (n_vars,)\n        n_vars: \u67d3\u8272\u4f53\u957f\u5ea6\n    Returns:\n        \u53d8\u5f02\u540e\u7684\u67d3\u8272\u4f53 (\u957f\u5ea6\u4e0e\u8f93\u5165\u76f8\u540c)\n    # \u6ce8\u610f\uff1a\n    # 1. solution \u662f\u4e00\u7ef4 numpy.ndarray\uff08\u957f\u5ea6 = n_vars\uff09\uff0c\u4e0d\u662f list\u3002\n    # 2. \u7981\u6b62\u4f7f\u7528 pop / append / remove / insert \u7b49\u5217\u8868\u64cd\u4f5c\u3002\n    # 3. \u5fc5\u987b\u4fdd\u8bc1\u8fd4\u56de\u503c\u4ecd\u662f numpy.ndarray\uff0c\u957f\u5ea6\u4e0d\u53d8\u3002\n    # 4. \u51fa\u9519\u65f6\u76f4\u63a5\u8fd4\u56de\u8f93\u5165\u7684\u62f7\u8d1d\uff0c\u4fdd\u8bc1\u8fdb\u5316\u4e0d\u4e2d\u65ad\u3002\n    """\n    try:\n        if n_vars < 2:\n            return solution.copy()\n        \n        mutated = solution.copy()\n        idx1, idx2 = np.random.choice(n_vars, 2, replace=False)\n        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]\n        return mutated\n        \n    except Exception:\n        return solution.copy()\n\n'


def convert_pf_to_json(pf_history):
    """将帕累托前沿历史数据转换为JSON可序列化的格式"""
    json_pf = []
    for run in pf_history:
        run_data = []
        for gen in run:
            if gen.size > 0:
                run_data.append(gen.tolist())
            else:
                run_data.append([])
        json_pf.append(run_data)
    return json_pf


###############################################################################

if __name__ == "__main__":
    # 0814 six_obj_exp: run_num:2, generation:50, pop_size=20,experiment_name:'six_obj_0815',six_obj_0815': 'PS_20250729081932'

    ##############################################
    # set parameters
    run_num = 1  # 实验次数 建议3-5
    generation = 30  # 进化代数 建议 50
    pop_size = 200  # 种群大小 对其eoh20

    ##############################################
    # get operators from outputs
    # default_operators_source = {
    #     "op_crossover": inspect.getsource(op_crossover),
    #     "op_mutation": inspect.getsource(op_mutation),
    #     "ma_crossover": inspect.getsource(ma_crossover),
    #     "ma_mutation": inspect.getsource(ma_mutation),
    # }
    # operators_functions = {
    #     "op_crossover": op_crossover,
    #     "op_mutation": op_mutation,
    #     "ma_crossover": ma_crossover,
    #     "ma_mutation": ma_mutation,
    # }
    # operators_functions = {
    #     "op_crossover": function_to_callable(op_crossover_json[0]),
    #     "op_mutation": function_to_callable(op_mutation_json[0]),
    #     "ma_crossover": function_to_callable(ma_crossover_json[0]),
    #     "ma_mutation": function_to_callable(ma_mutation_json[0]),
    # }
    # 评估算子
    experiment_name = "FJSP_OC_MDP_WinUCB2"
    operator_dir = "outputs/" + experiment_name + "/"
    json_file_path = os.path.join(operator_dir, "best_results_mk15.txt.json")

    # 从JSON文件加载最佳算子
    operators_functions = {}
    best_operators = load_best_operators_from_json(json_file_path)
    # for operator_code in best_operators.items():
    #     operator = function_string_to_program(operator_code, operator_name)
    operators_functions["op_crossover"] = function_string_to_program(
        best_operators["op_crossover"], "op_crossover"
    )
    operators_functions["op_mutation"] = function_string_to_program(
        best_operators["op_mutation"], "op_mutation"
    )
    operators_functions["op_crossover"] = function_string_to_program(
        best_operators["op_crossover"], "op_crossover"
    )
    operators_functions["op_crossover"] = function_string_to_program(
        best_operators["op_crossover"], "op_crossover"
    )
    # operators_functions = {
    #     "op_crossover": function_string_to_program(op_crossover_json, "op_crossover"),
    #     "op_mutation": function_string_to_program(op_mutation_json, "op_crossover"),
    #     "ma_crossover": function_string_to_program(ma_crossover_json, "op_crossover"),
    #     "ma_mutation": function_string_to_program(ma_mutation_json, "op_crossover"),
    # }
    # generate results save path
    experiment_name = "log"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = "WinUCB_30-200"
    output_folder = os.path.join("debug_alg", f"{experiment_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    ##############################################
    # set objectives and instances
    obj_list = ["makespan", "max_load"]
    # get test instances
    benchmark = "brandimarte"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instances_dir = os.path.join(BASE_DIR, "Instances", benchmark)
    instance_paths = glob.glob(os.path.join(instances_dir, "*.txt"))
    instance_paths.sort()
    # print instance paths
    print(f"Found {len(instance_paths)} instances in '{benchmark}' benchmark:")
    for i, path in enumerate(instance_paths):
        print(f"  {i+1}. {os.path.basename(path)}")

    # 添加收敛记录数据结构
    convergence_history = {
        "avg_hv": [],
        "max_hv": [],
        "min_obj1": [],
        "avg_obj1": [],
        "max_obj1": [],
        "min_obj2": [],
        "avg_obj2": [],
        "max_obj2": [],
        "pf": [],
    }
    for instance_path in instance_paths[14:15]:  # only test one instance
        instance_name = os.path.basename(instance_path)  # get instance name
        print(f"\n===== Evaluating instance: {instance_name} =====")
        problem = FJSP_Problem(obj_list, instance_path)  # initialize problem
        problem.ref_points = [[1498.2, 806.3]]  # fix the reference point
        # debug迭代多少次
        for i in range(1):
            # 编码方式
            Encoding = ["RI", "RI"]  # 两段都使用连续数值编码
            NIND = pop_size  # 种群规模

            # 创建字段信息
            Field_1 = ea.crtfld(
                Encoding[0],
                problem.varTypes[: problem.MS_len],
                problem.ranges[:, : problem.MS_len],
                problem.borders[:, : problem.MS_len],
            )
            Field_2 = ea.crtfld(
                Encoding[1],
                problem.varTypes[problem.MS_len :],
                problem.ranges[:, problem.MS_len :],
                problem.borders[:, problem.MS_len :],
            )
            # 创建种群
            population = ea.PsyPopulation(Encoding, [Field_1, Field_2], NIND)

            # 创建算法实例
            obj_num = len(problem.obj_list)
            myAlgorithm = NSGA_FJSP(problem, population, **operators_functions)

            # 添加收敛记录到算法
            myAlgorithm.convergence_history = copy.deepcopy(convergence_history)

            # 设置算法参数
            myAlgorithm.MAXGEN = generation
            myAlgorithm.logTras = 1
            myAlgorithm.verbose = True
            myAlgorithm.drawing = 0

            # 运行算法
            BestIndi, population, history = myAlgorithm.run()

            # 记录本次运行的收敛历史
            for key in convergence_history.keys():
                convergence_history[key].append(history[key])

            # 保存最终帕累托前沿
            final_pf = history["pf"][-1]

            # 选取makespan最小的前沿解保存甘特图
            best_idx = np.argmin(final_pf[:, 0])  # makespan最小
            best_chrom = population[best_idx].Phen  # 取对应染色体
            gantt_path = os.path.join(output_folder, f"gantt_run_{i+1}.png")
            # problem.save_gantt(best_chrom[0], gantt_path)

            np.savetxt(
                os.path.join(output_folder, f"final_pf_run_{i+1}.csv"),
                final_pf,
                delimiter=",",
                header="makespan,max_load",
                comments="",
            )

            print(f"Run {i+1} finished")

            fitness = BestIndi
            print(f"thread finished:")
            print(BestIndi)
    # 绘制收敛曲线
    import matplotlib.pyplot as plt

    # 计算多轮运行的平均值
    avg_hv = np.mean(convergence_history["avg_hv"], axis=0)
    max_hv = np.mean(convergence_history["max_hv"], axis=0)

    min_obj1 = np.mean(convergence_history["min_obj1"], axis=0)
    avg_obj1 = np.mean(convergence_history["avg_obj1"], axis=0)
    max_obj1 = np.mean(convergence_history["max_obj1"], axis=0)

    min_obj2 = np.mean(convergence_history["min_obj2"], axis=0)
    avg_obj2 = np.mean(convergence_history["avg_obj2"], axis=0)
    max_obj2 = np.mean(convergence_history["max_obj2"], axis=0)

    generations = np.arange(1, len(avg_hv) + 1)

    plt.figure(figsize=(15, 10))

    # HV收敛曲线
    plt.subplot(2, 2, 1)
    plt.plot(generations, avg_hv, "b-", label="Average HV")
    plt.plot(generations, max_hv, "r-", label="Max HV")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume (HV)")
    plt.title("HV Convergence")
    plt.legend()
    plt.grid(True)

    # 目标1（makespan）收敛曲线
    plt.subplot(2, 2, 2)
    plt.plot(generations, min_obj1, "g-", label="Min Makespan")
    plt.plot(generations, avg_obj1, "b-", label="Avg Makespan")
    plt.plot(generations, max_obj1, "r-", label="Max Makespan")
    plt.xlabel("Generation")
    plt.ylabel("Makespan")
    plt.title("Makespan Convergence")
    plt.legend()
    plt.grid(True)

    # 目标2（max_load）收敛曲线
    plt.subplot(2, 2, 3)
    plt.plot(generations, min_obj2, "g-", label="Min Max Load")
    plt.plot(generations, avg_obj2, "b-", label="Avg Max Load")
    plt.plot(generations, max_obj2, "r-", label="Max Max Load")
    plt.xlabel("Generation")
    plt.ylabel("Max Load")
    plt.title("Max Load Convergence")
    plt.legend()
    plt.grid(True)

    # 种群多样性分析（最终帕累托前沿）
    plt.subplot(2, 2, 4)

    # 使用不同颜色区分不同运行
    colors = plt.cm.tab10.colors

    for run_idx in range(run_num):
        # 获取最终代的帕累托前沿
        final_gen_idx = len(convergence_history["pf"][run_idx]) - 1
        pf = convergence_history["pf"][run_idx][final_gen_idx]

        if len(pf) > 0:
            plt.scatter(
                pf[:, 0],
                pf[:, 1],
                s=20,
                alpha=0.7,
                c=[colors[run_idx % len(colors)]],
                label=f"Run {run_idx+1}",
            )

    plt.xlabel("Makespan")
    plt.ylabel("Max Load")
    plt.title("Final Pareto Fronts (Diversity Analysis)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "convergence_and_diversity.png"))
    plt.show()

    # 保存所有收敛历史数据为JSON
    convergence_json = {
        "avg_hv": convergence_history["avg_hv"],
        "max_hv": convergence_history["max_hv"],
        "min_obj1": convergence_history["min_obj1"],
        "avg_obj1": convergence_history["avg_obj1"],
        "max_obj1": convergence_history["max_obj1"],
        "min_obj2": convergence_history["min_obj2"],
        "avg_obj2": convergence_history["avg_obj2"],
        "max_obj2": convergence_history["max_obj2"],
        "pf": convert_pf_to_json(convergence_history["pf"]),
    }
    with open(os.path.join(output_folder, "convergence_history.json"), "w") as f:
        json.dump(convergence_json, f, indent=2)

    print(
        f"收敛历史数据已保存为: {os.path.join(output_folder, 'convergence_history.json')}"
    )
