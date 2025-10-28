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
from Problem.FJSP.FJSP_Problem import FJSP_Problem
from Algrithm.NSGA3_newcode import NSGA_FJSP
from llm4ad.base import Evaluation
from utils.code_json_to_program import function_json_to_program
from _evaluate import (
    ma_crossover,
    ma_mutation,
    op_crossover,
    op_mutation,
)

import geatpy as ea
import numpy as np


###############################################################################

if __name__ == "__main__":
    # 0814 six_obj_exp: run_num:2, generation:50, pop_size=20,experiment_name:'six_obj_0815',six_obj_0815': 'PS_20250729081932'

    ##############################################
    # set parameters
    run_num = 1  # 实验次数 建议3-5
    generation = 50  # 进化代数 建议 50
    pop_size = 200  # 种群大小 对其eoh20

    ##############################################
    # get operators from outputs
    operators = {
        "op_crossover": op_crossover,
        "op_mutation": op_mutation,
        "ma_crossover": ma_crossover,
        "ma_mutation": ma_mutation,
    }
    # generate results save path
    experiment_name = "log"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = 1
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
    for instance_path in instance_paths[:1]:  # only test one instance
        instance_name = os.path.basename(instance_path)  # get instance name
        print(f"\n===== Evaluating instance: {instance_name} =====")
        problem = FJSP_Problem(obj_list, instance_path)  # initialize problem
        # problem.ref_points = [[103.0, 103.0]]  # fix the reference point
        # debug迭代多少次
        for i in range(10):
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
            if obj_num == 1:
                myAlgorithm = ea.soea_psy_EGA_templet(problem, population)
            elif obj_num > 1:
                myAlgorithm = algorithm(problem, population, **operators)

            # 设置算法参数
            myAlgorithm.MAXGEN = generation_num
            myAlgorithm.logTras = 1
            myAlgorithm.verbose = True
            myAlgorithm.drawing = 0

            # 运行算法
            [BestIndi, population] = myAlgorithm.run()

            # 计算HV指标
            # hv = ea.indicator.HV(BestIndi.ObjV, PF=ref_point)

            # 返回完整结果
            # return resulst_calculation(pareto_front, hv, generation_num, pop_size)
            # 获取帕累托前沿
            fitness = np.array(BestIndi.ObjV).tolist()
            print(f"thread finished:")
            print(BestIndi.ObjV)
