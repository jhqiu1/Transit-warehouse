import os
import random
from typing import List, Any

import numpy as np
import geatpy as ea

from aad_catl_search_operators.CATL_LLM.MyProblem import MyProblem  # 导入自定义问题接口

# from aad_catl_search_operators.CATL.localDataProcess import localDataProcess
from aad_catl_search_operators.utils.path_util import get_path
from NSGA3_LLM import nsga_llm
from llm4ad.base import Evaluation


#######################################################################
#### todo: 自定义：main_op调用算子进行计算，并返回最终得分
def main_op(
    ref_point=np.array([[16000.0, 280.0]]),
    algorithm=nsga_llm,
    problem=None,
    generation_num=20,
    pop_size=20,
    operator: callable = None,
    operator_name=None,
    process_and_return: callable = None,
):
    # todo: 重要参数：operator：大模型搜到算子

    Encoding = ["RI", "P"]
    # 编码方式
    NIND = pop_size  # 种群规模
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
    population = ea.PsyPopulation(Encoding, [Field_1, Field_2], NIND)

    # todo: 使用大模型搜到算子：将operator传入algorithm，替换原本算子
    obj_num = len(problem.obj_list)
    if obj_num == 1:
        myAlgorithm = ea.soea_psy_EGA_templet(problem, population)  #
    elif obj_num > 1:
        kwargs = {operator_name: operator}
        myAlgorithm = algorithm(problem, population, **kwargs)
    myAlgorithm.MAXGEN = generation_num
    myAlgorithm.logTras = 1
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 0

    [BestIndi, population] = myAlgorithm.run()

    if process_and_return is not None:
        result = process_and_return(problem, BestIndi)
        return result

    # todo: 计算评估指标。如果指标依赖于某些常量（如ref_point），则作为输入参数，保持搜索过程中一致 (HV计算方式)
    hv = ea.indicator.HV(BestIndi.ObjV, PF=ref_point)
    # 加权方式 实际obj/ref point  + hv
    # 可以考虑给归一化后 加权 保证目标优先级
    print(f"hv: {hv}")
    return hv


def generate_vectors(problem, num_vectors=10):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    N = problem.Dims
    lbs = problem.lbs
    ubs = problem.ubs + 1

    job_num = problem.Dims - problem.MS_len

    results = []

    for i in range(num_vectors):
        # 生成随机向量
        vector = np.random.randint(lbs, ubs, size=N)

        # 生成 [i for i in range(25)] 的随机排列
        random_permutation = np.random.permutation(job_num)

        # 覆盖向量的后 25 位
        vector[-job_num:] = random_permutation

        # 计算目标函数值
        obj_values = problem.calcFunc(vector)
        obj_values = [obj_values[obj] for obj in problem.obj_list]
        results.append(obj_values)
    return results


def generate_ref_point(problem, num_vectors=10):
    fitness_list = generate_vectors(problem, num_vectors)
    ref_point = np.zeros(len(fitness_list[0]))
    for i in range(len(ref_point)):
        ref_point[i] = max([fitness[i] for fitness in fitness_list]) * 2
    # 临时取一个最大值，作为参考点
    # ref_point = np.array([[10000,10000]]).astype('float')
    ref_point = np.array([[float(v) for v in ref_point]])
    return ref_point


def evaluate(
    ref_point,
    algorithm,
    problem,
    operator_name,
    operator,
    n_evals: int = 11,
    generation_num=20,
    ret_all_results=False,
):
    """
    :param algorithm:   指定使用的算法框架。必填
    :param obj_list:    指定优化目标列表。必填
    :param operator:    operator将被传入algorithm。必填
    :param n_evals:     每次评估的重复次数。选填
    :param generation_num:      每次评估的运行代数。建议填
    :return:
    """
    # data = localDataProcess_test(if_all_peocess=True)

    all_res = []
    for i in range(n_evals):
        # randomly setup random seeds
        np.random.seed()
        random.seed()
        res = main_op(
            ref_point,
            algorithm,
            problem,
            operator_name=operator_name,
            operator=operator,
            pop_size=20,
            generation_num=generation_num,
        )
        all_res.append(res)
    if ret_all_results:
        return all_res
    else:
        return np.mean(all_res)


########################################################################
#### todo: 自定义：MyEvaluation调用main_op获得算子得分
class MyEvaluation(Evaluation):
    def __init__(
        self,
        template,
        problem,
        algorithm,
        ref_point,
        operator_name,
        generation_num=20,
        n_evals=6,
        ret_all_results=False,
    ):
        super().__init__(template_program=template, timeout_seconds=1500)
        self.n_evals = n_evals
        self.ret_all_results = ret_all_results
        self.debug_mode = True
        self.algorithm = algorithm
        self.generation_num = generation_num
        self.problem = problem
        self.ref_point = ref_point
        self.operator_name = operator_name

    # todo: 评估callable_func，并返回最终得分
    def evaluate_program(
        self, program_str: str, callable_func: callable, **kwargs
    ) -> Any | None:
        res = evaluate(
            algorithm=self.algorithm,
            ref_point=self.ref_point,
            problem=self.problem,
            generation_num=self.generation_num,
            operator=callable_func,
            n_evals=self.n_evals,
            ret_all_results=self.ret_all_results,
            operator_name=self.operator_name,
        )
        print(f"Result Mean: {np.mean(res)}")
        print(f"Result Standard Deviation: {np.std(res)}")
        res = np.mean(res)
        return res


perm_mutation_template = '''
import numpy as np
from typing import List

def permutation_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    """ permutation local search operator
    Args:
        solution    : an individual solution with shape=(n_vars,). solution has `n_vars` integer variables.
                      Please note that solution is a permutation, e.g. np.ndarray([3, 2, 0, 4, 1]) is a permutation with `n_vars` = 5
        n_vars      : number of variables for each individual solution. n_vars = len(solution)
    Returns:
        A new permutation solution with shape=(n_vars,)
    """
    import random
    i, j = random.sample(range(len(solution)), 2)
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution
'''

assign_mutation_template = '''
import numpy as np
from typing import List

def assignment_mutation(X: np.ndarray, n_vars: int, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    """Mutation operator.
    Args:
        X       : an individual solution with shape=(n_vars,). X has `n_vars` integer variables.
                  Please note that xl[i] <= X[i] <= xu[i].
        n_vars  : number of variables for each individual.
        xl      : shape=(n_vars,), xl[i] refers to the lower bound for the i-th variable for X.
        xu      : shape=(n_vars,), xu[i] refers to the upper bound for the i-th variable for X.
    Returns:
        mutated solutions, which shape=(n_vars,).
    """

    mutation_prob = np.random.rand(n_vars)
    mutation_mask = mutation_prob < (1 / n_vars)
    DisI = 20

    for i in range(n_vars):
        if mutation_mask[i]:
            delta = np.random.rand() ** (1 / (DisI + 1)) - 1
            X[i] += delta * (xu[i] - xl[i])
        X[i] = min(max(round(X[i]), xl[i]), xu[i])
    return X
'''

perm_crossover_template = '''
import numpy as np
from typing import List

def permutation_crossover(population: np.ndarray, N: int, D: int) -> np.ndarray:
    """permutation_crossover
    Args:
        population       : a 2-dimension ndarray with shape=(N, D). Population has`N` individuals with shape = (D,). Each individual is a permutation, e.g. np.ndarray([3, 2, 0, 4, 1]) is a permutation with `D` = 5.
        N  : number of individuals in the population.
        D  : number of variables for each individual. 
    Returns:
        a new population, which shape=(N, D).
        Please make sure that each individual in the new population must be a legit permutation. For example, np.ndarray([3, 2, 0, 4, 1]) is a legit permutation with `D` = 5, but np.ndarray([3, 2, 0, 5, 1]) is not a legit permutation.
    """
    new_population = population.copy()

    for i in range(0, N - 1, 2):
        if np.random.rand() < 0.7:
            # 将父代转换为列表，便于操作
            parent1 = population[i].copy().tolist()
            parent2 = population[i + 1].copy().tolist()

            # 随机选取两个交叉点
            cxpoint1 = np.random.randint(0, D)
            cxpoint2 = np.random.randint(0, D)
            if cxpoint1 > cxpoint2:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            # 建立映射关系
            # mapping1：用于 child2 的冲突解决，映射关系为 parent1 -> parent2
            # mapping2：用于 child1 的冲突解决，映射关系为 parent2 -> parent1
            mapping1 = {}
            mapping2 = {}
            for j in range(cxpoint1, cxpoint2 + 1):
                mapping1[parent1[j]] = parent2[j]
                mapping2[parent2[j]] = parent1[j]

            child1 = [None] * D
            child2 = [None] * D

            # 将交叉区间直接复制到子代中
            child1[cxpoint1:cxpoint2 + 1] = parent2[cxpoint1:cxpoint2 + 1]
            child2[cxpoint1:cxpoint2 + 1] = parent1[cxpoint1:cxpoint2 + 1]

            # 辅助函数：解决冲突，防止映射循环（记录已访问的基因）
            def resolve(mapping, gene):
                visited = set()
                while gene in mapping:
                    if gene in visited:
                        # 如果存在循环，直接退出循环返回 gene（或根据需求处理）
                        raise ValueError('Duplicate gene: {}'.format(gene))
                    visited.add(gene)
                    gene = mapping[gene]
                return gene

            # 填充非交叉区间
            for j in list(range(0, cxpoint1)) + list(range(cxpoint2 + 1, D)):
                # child1 使用 parent1 的基因，但若冲突则用 mapping2（映射 parent2->parent1）
                gene = parent1[j]
                if gene in child1[cxpoint1:cxpoint2 + 1]:
                    gene = resolve(mapping2, gene)
                child1[j] = gene

                # child2 使用 parent2 的基因，但若冲突则用 mapping1（映射 parent1->parent2）
                gene = parent2[j]
                if gene in child2[cxpoint1:cxpoint2 + 1]:
                    gene = resolve(mapping1, gene)
                child2[j] = gene

            new_population[i] = child1
            new_population[i + 1] = child2

    return new_population
'''

assign_crossover_template = '''
import numpy as np
from typing import List

def assignment_crossover(population: np.ndarray, N: int, D: int) -> np.ndarray:
    """assignment_crossover
    Args:
        population       : a 2-dimension ndarray with shape=(N, D). Population has`N` individuals with shape = (D,).
        N  : number of individuals in the population.
        D  : number of variables for each individual.
    Returns:
        a new population, which shape=(N, D).
    """
    new_population = population.copy()

    for i in range(0, N - 1, 2):
        if np.random.rand() < 0.7:
            a, b = sorted(np.random.choice(D, 2, replace=False))  # 随机选一个区间 [a:b]
            new_population[i, a:b], new_population[i + 1, a:b] = population[i + 1, a:b], population[i, a:b]  # 交换该片段
    return new_population
'''

if __name__ == "__main__":

    ###################################################################

    def assignment_crossover(population: np.ndarray, N: int, D: int) -> np.ndarray:
        """assignment_crossover
        Args:
            population       : a 2-dimension ndarray with shape=(N, D). Population has`N` individuals with shape = (D,).
            N  : number of individuals in the population.
            D  : number of variables for each individual.
            xl : shape=(D,), xl[i] refers to the lower bound for the i-th variable for each individual.
            xu : shape=(D,), xu[i] refers to the upper bound for the i-th variable for each individual.
        Returns:
            a new population, which shape=(N, D).
        """
        new_population = population.copy()

        for i in range(0, N - 1, 2):
            if np.random.rand() < 0.7:
                a, b = sorted(np.random.choice(D, 2, replace=False))
                new_population[i, a:b], new_population[i + 1, a:b] = (
                    population[i + 1, a:b],
                    population[i, a:b],
                )
        return new_population

    def assignment_mutation(
        X: np.ndarray, n_vars: int, xl: List[int], xu: List[int]
    ) -> np.ndarray:
        """Mutation operator.
        Args:
            X       : an individual solution with shape=(n_vars,). X has `n_vars` integer variables.
                      Please note that xl[i] <= X[i] <= xu[i].
            n_vars  : number of variables for each individual.
            xl      : shape=(n_vars,), xl[i] refers to the lower bound for the i-th variable for X.
            xu      : shape=(n_vars,), xu[i] refers to the upper bound for the i-th variable for X.
        Returns:
            mutated solutions, which shape=(n_vars,).
        """

        mutation_prob = np.random.rand(n_vars)
        mutation_mask = mutation_prob < (1 / n_vars)
        DisI = 20

        for i in range(n_vars):
            if mutation_mask[i]:
                delta = np.random.rand() ** (1 / (DisI + 1)) - 1
                X[i] += delta * (xu[i] - xl[i])
            X[i] = min(max(round(X[i]), xl[i]), xu[i])
        return X

    # data = localDataProcess(get_path('aad_catl_search_operators/data/train.xlsx'), if_all_peocess=True)
    obj_list = ["obj_1", "obj_3"]

    batchNo = "PS_20250729081932"
    problem = MyProblem(obj_list, batchNo)
    ref_point = generate_ref_point(problem, 10)

    print(f"ref_point: {ref_point}")

    evaluator = MyEvaluation(
        template=assign_crossover_template,
        operator_name="assignment_crossover",
        algorithm=nsga_llm,
        problem=problem,
        ref_point=ref_point,
        generation_num=3,
        n_evals=1,
    )

    evaluator.evaluate_program("test", assignment_crossover)
