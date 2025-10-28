import sys
import os


import os
import json
import copy
import random
import time

import sys

import concurrent.futures
from copy import deepcopy
from aad_catl_search_operators.utils.code_json_to_program import (
    function_json_to_program,
)
from aad_catl_search_operators.CATL_LLM.NSGA3_LLM import nsga_llm
from aad_catl_search_operators.CATL_LLM.MyProblem import MyProblem  # 导入自定义问题接口

# from aad_catl_search_operators.CATL.localDataProcess import localDataProcess
from aad_catl_search_operators.utils.path_util import get_path

# from aad_catl_search_operators.CATL_LLM._evaluate import generate_ref_point

from llm4ad.base import Evaluation

import geatpy as ea
import numpy as np


###############################################################################

###############################################################################


def main_pls(
    problem,
    pop_size,
    generation_num,
    operator_path,
    operator_name=None,
    algorithm=nsga_llm,
    process_and_return=None,
):
    """
    Args:
        perm_operator_path: 项目顺序算子路径
        assign_operator_path: 设备选择算子路径
    Returns:
        找到优解的目标函数值
    """

    Encoding = ["RI", "P"]
    # 编码方式
    NIND = pop_size  # 种群规模
    # Field_1 = ea.crtfld(Encoding[0], problem.varTypes[:data.MS_len], problem.ranges[:, :data.MS_len],
    #                     problem.borders[:, :data.MS_len])
    # Field_2 = ea.crtfld(Encoding[1], problem.varTypes[data.MS_len:], problem.ranges[:, data.MS_len:],
    #                     problem.borders[:, data.MS_len:])
    Field_1 = ea.crtfld(
        Encoding[0],
        problem.varTypes[: problem.MS_len],
        problem.ranges[:, : problem.MS_len],
        problem.borders[:, : problem.MS_len],
    )  # 创建区域描述器
    Field_2 = ea.crtfld(
        Encoding[1],
        problem.varTypes[problem.MS_len :],
        problem.ranges[:, problem.MS_len :],
        problem.borders[:, problem.MS_len :],
    )
    population = ea.PsyPopulation(Encoding, [Field_1, Field_2], NIND)

    # todo: 使用大模型搜到算子：将operator传入algorithm，替换原本算子
    obj_num = len(problem.obj_list)
    if obj_num == 1:
        myAlgorithm = ea.soea_psy_EGA_templet(problem, population)  #
    elif obj_num > 1:
        if operator_name is not None and operator_path is not None:
            operator = function_json_to_program(operator_path)
            kwargs = {operator_name: operator}
            myAlgorithm = algorithm(problem, population, **kwargs)
        else:
            myAlgorithm = algorithm(problem, population)
    myAlgorithm.MAXGEN = generation_num
    myAlgorithm.logTras = 1
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 0

    [BestIndi, population] = (
        myAlgorithm.run()
    )  # bestindi 按字典序排序 第一个为最优 升序

    if process_and_return is not None:
        result = process_and_return(problem, BestIndi)
        return result

    fitness = np.array(BestIndi.ObjV).tolist()
    print(f"thread finished:")
    print(BestIndi.ObjV)
    # print(fitness)
    return fitness
    # print(ref_point)
    # print(BestIndi.ObjV)
    # # todo: 计算评估指标。如果指标依赖于某些常量（如ref_point），则作为输入参数，保持搜索过程中一致
    # hv = ea.indicator.HV(np.array(BestIndi.ObjV).astype('float'), PF=ref_point)
    # print(f"hv: {hv}")
    # return hv


def run_single(i, main_args, folder_name):
    args_copy = deepcopy(main_args)

    os.makedirs(os.path.join(folder_name, "profile"), exist_ok=True)

    # profiler = cProfile.Profile()
    # profiler.enable()

    start = time.time()
    result_dict = {"F": [main_pls(**args_copy)]}
    end = time.time()

    # profiler.disable()
    # with open(f'{folder_name}/profile/run_{i}.txt', 'w') as f:
    #     stats = pstats.Stats(profiler, stream=f)
    #     stats.strip_dirs().sort_stats('tottime').print_stats(30)

    result_dict["time"] = end - start
    with open(f"{folder_name}/run_{i}.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    print(f"finished: run_{i}")
    return f"run_{i}", result_dict


# 多进程评估算子
def run_instance(main_args, folder_name, run_num=5):
    os.makedirs(folder_name, exist_ok=True)

    benchmark_dict = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single, i, main_args, folder_name)
            for i in range(run_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            run_id, result = future.result()
            benchmark_dict[run_id] = result

    return benchmark_dict


def run_baseline(
    main_args,
    folder_name,
    run_num=3,
):
    os.makedirs(folder_name, exist_ok=True)
    start = time.time()
    benchmark_result = run_instance(main_args, folder_name, run_num=run_num)
    finish = time.time()

    main_args["obj_list"] = problem.obj_list
    main_args.pop("algorithm", None)
    main_args.pop("problem", None)

    # ref_point = main_args.pop('ref_point', None)
    # ref_point = ref_point.tolist()
    # main_args['ref_point'] = ref_point

    # main_args['function_eval'] = 2 * main_args['round'] * algorithm.termination_criteria.max_eval

    # main_args.pop('perm_operator', None)
    # main_args.pop('assign_operator', None)

    json_result = {
        # "data_args": {"data_path": problem._data.data_path},
        "main_args": main_args,
        "time": finish - start,
        "benchmark_result": benchmark_result,
    }
    with open(f"{folder_name}/result_baseline.json", "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=4)


# def generate_ref_point(problem, num_vectors=10):
#     fitness_list = generate_vectors(problem, num_vectors)
#     ref_point = np.zeros(len(fitness_list[0]))
#     for i in range(len(ref_point)):
#         ref_point[i] = max([fitness[i] for fitness in fitness_list]) * 2
#     # 临时取一个最大值，作为参考点
#     # ref_point = np.array([[10000,10000]]).astype('float')
#     ref_point = p.array([[float(v) for v in ref_point]])
#     return ref_point

if __name__ == "__main__":
    # 0814 six_obj_exp: run_num:2, generation:50, pop_size=20,experiment_name:'six_obj_0815',six_obj_0815': 'PS_20250729081932'

    ##############################################
    # todo: 自定义超参
    run_num = 2  # 实验次数 建议3-5
    generation = 5  # 进化代数 建议 50
    pop_size = 5  # 种群大小 对其eoh20

    ##############################################
    experiment_name = "six_obj"

    ##############################################
    # todo: 导入搜到算子
    operator_folder_path = (
        "aad_catl_search_operators/CATL_LLM/six_obj/{}"  # todo: 算子结果路径
    )
    operators_path = {
        # '': None,
        "permutation_crossover": get_path(
            operator_folder_path.format("permutation_crossover")
        ),
        "permutation_mutation": get_path(
            operator_folder_path.format("permutation_mutation")
        ),
        "assignment_crossover": get_path(
            operator_folder_path.format("assignment_crossover")
        ),
        "assignment_mutation": get_path(
            operator_folder_path.format("assignment_mutation")
        ),
    }

    ##############################################
    # todo: 定义problem
    obj_list = ["obj_1", "obj_2", "obj_3", "obj_4", "obj_5", "obj_6"]
    data_dict = {"six_obj_0815": "PS_20250812141441"}  # PS_20250814135919'

    # ref_point = np.array([1e12, 1000]).astype('float')
    # 训练数据集 PS_20250716203528 ：10个pack
    # 验证数据集 PS_20250717142352，9个pack

    # todo: 定义输出路径
    output_folder = str(get_path("aad_catl_search_operators/results"))

    # data_dict = {
    #     'train': batch_train,
    #     'test_10': batch_test_10,
    #     # 'test_20': batch_test_20
    # }

    for data_name, batchNo in data_dict.items():

        problem = MyProblem(obj_list, batchNo, output_folder)
        # ref_point = generate_ref_point(problem, data, 10)
        # from _evaluate import generate_ref_point

        # ref_point = generate_ref_point(problem, 10)

        print(f"MS_len: {problem.MS_len}")
        print(f"Var len: {problem.Dim}")

        for operator, operator_path in operators_path.items():

            main_args = {
                "problem": problem,
                "pop_size": pop_size,
                # 'ref_point': None,
                "generation_num": generation,
                "algorithm": nsga_llm,
                "operator_name": operator,
                "operator_path": operator_path,
            }

            if operator == "":
                folder_name = f"{output_folder}/{experiment_name}/{data_name}/NSGA"
            else:
                folder_name = (
                    f"{output_folder}/{experiment_name}/{data_name}/NSGA+{operator}"
                )

            run_baseline(main_args, run_num=run_num, folder_name=folder_name)
