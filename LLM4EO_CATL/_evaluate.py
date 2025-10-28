import os
import glob
import json
import numpy as np
from typing import List, Dict, Callable, Optional, Any, Union, Tuple
from Problem.FJSP.FJSP_Problem import FJSP_Problem
from Algrithm.NSGA2 import NSGA_FJSP

# from _storage import GlobalResultStorage
from Logger import Logger
import random
import geatpy as ea
import threading
from copy import deepcopy

from llm4ad.base import Evaluation


# 转换为可调用函数
def string_to_callable(function_string, function_name=None):
    """
    将函数定义字符串转换为可调用函数对象

    参数:
        function_string: 包含函数定义的字符串
        function_name: 可选，要提取的函数名称。如果未提供，则从字符串中自动提取

    返回:
        可调用的函数对象，如果转换失败则返回None
    """
    try:
        # 创建新的命名空间并导入必要的模块
        namespace = {}

        # 导入代码中可能需要的常用模块
        try:
            import numpy as np

            namespace["np"] = np
        except ImportError:
            print("Warning: numpy not available")

        try:
            from typing import Tuple, List, Dict, Any, Union, Optional

            namespace["Tuple"] = Tuple
            namespace["List"] = List
            namespace["Dict"] = Dict
            namespace["Any"] = Any
            namespace["Union"] = Union
            namespace["Optional"] = Optional
        except ImportError:
            print("Warning: typing module not available")

        # 执行代码字符串
        exec(function_string, namespace)

        # 提取函数名（如果未提供）
        if function_name is None:
            # 从字符串中提取函数名
            lines = function_string.strip().split("\n")
            for line in lines:
                if line.startswith("def "):
                    function_name = line.split("def ")[1].split("(")[0].strip()
                    break

        # 获取函数引用
        callable_func = namespace.get(function_name)

        if callable_func is not None and callable(callable_func):
            return callable_func
        else:
            print(
                f"Warning: Function '{function_name}' not found or not callable in code"
            )
            return None

    except Exception as e:
        print(f"Error converting string to callable: {e}")
        return None


def main_op(
    ref_point: np.ndarray,
    algorithm: Callable,
    problem: FJSP_Problem,
    generation_num: int,
    pop_size: int,
    operators: Dict[str, Callable],
    process_and_return: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    执行优化算法并返回完整结果

    参数:
        ref_point: 参考点 (用于计算HV指标)
        algorithm: 算法类
        problem: 问题实例
        generation_num: 进化代数
        pop_size: 种群大小
        operators: 算子字典 (包含四个算子)
        process_and_return: 结果处理函数 (可选)

    返回:
        包含完整结果的字典
    """
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
        myAlgorithm = ea.soea_psy_EGA_templet(problem, population)  # single objective
    elif obj_num > 1:
        myAlgorithm = algorithm(problem, population, **operators)

    # 设置算法参数
    myAlgorithm.MAXGEN = generation_num
    myAlgorithm.logTras = 1
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 0
    # 运行算法
    # BestIndi, resultOr = myAlgorithm.run()
    # 获取帕累托前沿
    # pareto_front = np.array([ind.ObjV for ind in BestIndi], dtype=float).reshape(-1, 2)
    # # 运行算法
    [BestIndi, population], resultOr = myAlgorithm.run()

    # # 获取帕累托前沿
    pareto_front = BestIndi.ObjV

    # 计算HV指标
    if resultOr:
        hv = ea.indicator.HV(pareto_front, PF=np.array(ref_point))
    else:
        hv = 0
    # 返回完整结果
    return resulst_calculation(pareto_front, hv, generation_num, pop_size)


# 计算指标
def resulst_calculation(pareto_front, hv, generation_num, pop_size):
    num_solutions = len(pareto_front)
    avg_makespan = np.mean(pareto_front[:, 0])
    avg_utilization = np.mean(pareto_front[:, 1])
    min_makespan = np.min(pareto_front[:, 0])
    max_utilization = np.max(pareto_front[:, 1])
    return {
        "hv": hv,
        "pareto_front": pareto_front,
        "makespan": min_makespan,
        "utilization": max_utilization,
        "generations": generation_num,
        "population_size": pop_size,
        "num_solutions": num_solutions,
        "avg_makespan": avg_makespan,
        "avg_utilization": avg_utilization,
        "min_makespan": min_makespan,
        "max_utilization": max_utilization,
    }


def evaluate(
    ref_point: np.ndarray,
    algorithm: Callable,
    problem: FJSP_Problem,
    operators: Dict[str, Callable],
    logger: Logger,
    n_evals: int = 11,
    generation_num: int = 20,
    ret_all_results: bool = False,
    pop_size: int = 200,
) -> Any:
    """
    多次评估算子组合的性能

    参数:
        ref_point: 参考点
        algorithm: 算法类
        problem: 问题实例
        operators: 算子字典 (包含四个算子)
        logger: 日志记录器实例
        n_evals: 评估次数
        generation_num: 进化代数
        ret_all_results: 是否返回所有结果

    返回:
        平均HV值或所有结果
    """
    # # 记录算例开始
    # logger.log_info(f"===== Evaluating instance: {problem.batchNo} =====")

    all_res = []
    for i in range(n_evals):
        # 设置随机种子
        np.random.seed()
        random.seed()

        # 记录运行开始
        # logger.log_run_start(i, n_evals)

        # 执行优化
        res = main_op(
            ref_point=ref_point,
            algorithm=algorithm,
            problem=problem,
            generation_num=generation_num,
            pop_size=pop_size,
            operators=operators,
        )
        all_res.append(res)

        # # 记录结果
        # logger.log_result(
        #     i,
        #     {
        #         "instance": problem.batchNo,
        #         "operator_name": operators.get("operator_name", "default"),
        #         "run_id": i,
        #         "hv": res["hv"],
        #         "makespan": res["makespan"],
        #         "utilization": res["utilization"],
        #         "population_size": pop_size,
        #         "generations": generation_num,
        #         "seed": i,
        #         "num_solutions": res["num_solutions"],
        #         "avg_makespan": res["avg_makespan"],
        #         "avg_utilization": res["avg_utilization"],
        #     },
        # )

        # # 记录帕累托前沿
        # logger.log_pareto_front(
        #     f"{problem.batchNo}_{operators.get('operator_name', 'default')}",
        #     res["pareto_front"],
        # )

        # # 记录运行完成
        # logger.log_run_complete(
        #     run_id=i, total_runs=n_evals, hv=res["hv"], pareto_front=res["pareto_front"]
        # )

    # 记录算例完成
    avg_hv = np.mean([r["hv"] for r in all_res])
    # 校验hv，异常则报error
    if np.isnan(avg_hv) or avg_hv < 0:
        raise ValueError(f"Invalid HV value: {avg_hv}")
    # logger.log_info(
    #     f"===== Instance {problem.batchNo} evaluation completed, Avg HV: {avg_hv:.4f} ====="
    # )

    if ret_all_results:
        return all_res
    else:
        return avg_hv


class MyEvaluation(Evaluation):
    def __init__(
        self,
        algorithm,
        problem,
        exp_name,
        template,
        ev_operator_name,
        ref_point,
        generation_num=20,
        n_evals=6,
        pop_size=200,
        ret_all_results=False,
    ):
        """
        初始化评估器

        参数:
            algorithm: 算法类
            exp_name: 实验名称
            ref_point: 参考点 (可选)
            generation_num: 进化代数
            n_evals: 评估次数
            ret_all_results: 是否返回所有结果
        """

        super().__init__(template_program=template, timeout_seconds=1500)
        self.n_evals = n_evals
        self.ret_all_results = ret_all_results
        self.algorithm = algorithm
        self.problem = problem
        self.generation_num = generation_num
        self.ref_point = ref_point
        self.exp_name = exp_name
        self.pop_size = pop_size
        self.ev_operator_name = ev_operator_name
        self.operators = {
            "op_crossover": None,
            "op_mutation": None,
            "ma_crossover": None,
            "ma_mutation": None,
        }

        # 本地历史记录和最佳结果追踪
        self.evaluation_history = []  # 存储 (算子组合, 分数) 元组
        self.best_score = float("-inf")
        self.best_operators = {}  # 存储最佳算子组合

        # 添加计数器用于调试
        self.evaluation_count = 0

        # # 创建顶层日志记录器
        # self.logger = Logger(exp_name=self.exp_name, output_dir="experiments")

    def set_operator(self, operator_name: str, operator: Callable):
        """设置算子"""
        if operator_name in self.operators:
            self.operators[operator_name] = operator
        else:
            raise ValueError(f"Invalid operator name: {operator_name}")

    def save_operatorstr_bydict(self, operators: Dict[str, Union[Callable, str]]):
        # 保存算子字符串
        if operators:
            self.oprerators_str = operators
            # for op_name, op_func in operators.items():
            #     if isinstance(op_func, str):
            #         # 如果是字符串，编译为函数对象
            #         callable_func = string_to_callable(op_func, op_name)
            #         if callable_func is not None:
            #             self.set_operator(op_name, callable_func)
            #         else:
            #             print(f"Warning: Failed to compile {op_name} from source code")
            #     else:
            #         # 如果是函数对象，直接设置
            #         self.set_operator(op_name, op_func)

    # todo: 评估callable_func，并返回最终得分, 供eoh调用
    def evaluate_program(
        self, program_str: str, new_callable_func: callable, **kwargs
    ) -> Union[Any, None]:
        # generate the evo-operator dict
        self.operators[self.ev_operator_name] = new_callable_func
        if self.oprerators_str:
            for op_name, op_func in self.oprerators_str.items():
                if (
                    op_name != self.ev_operator_name
                ):  # ev_operator_name is used llm operators
                    if isinstance(op_func, str):
                        # 如果是字符串，编译为函数对象
                        callable_func = string_to_callable(op_func, op_name)
                        if callable_func is not None:
                            self.set_operator(op_name, callable_func)
                        else:
                            print(
                                f"Warning: Failed to compile {op_name} from source code"
                            )
                    else:
                        # 如果是函数对象，直接设置
                        self.set_operator(op_name, op_func)

        # 设置参考点（如果未设置）
        if self.ref_point is None:
            self.ref_point = self.problem.ref_points
        print("waitting for evaluate...")
        # 评估
        res = evaluate(
            ref_point=self.ref_point,
            algorithm=self.algorithm,
            problem=self.problem,
            operators=self.operators,
            logger=None,
            n_evals=self.n_evals,
            generation_num=self.generation_num,
            ret_all_results=self.ret_all_results,
            pop_size=self.pop_size,
        )

        print(f"Result Mean: {res}")
        print(f"Result Standard Deviation: {res}")
        # self.record_evaluation(res)
        return res

    def evaluate_combination(self, operators: Dict[str, Union[Callable, str]]) -> float:
        """
        评估特定算子组合的性能

        参数:
            operators: 算子字典 (包含四个算子)，可以是函数对象或源代码字符串

        返回:
            平均HV值
        """
        # 设置算子
        if operators:
            for op_name, op_func in operators.items():
                if isinstance(op_func, str):
                    # 如果是字符串，编译为函数对象
                    callable_func = string_to_callable(op_func, op_name)
                    if callable_func is not None:
                        self.set_operator(op_name, callable_func)
                    else:
                        print(f"Warning: Failed to compile {op_name} from source code")
                else:
                    # 如果是函数对象，直接设置
                    self.set_operator(op_name, op_func)

        # 调用evaluate计算hv
        all_res = evaluate(
            ref_point=self.ref_point,
            algorithm=self.algorithm,
            problem=self.problem,
            operators=self.operators,
            logger=None,
            n_evals=self.n_evals,
            generation_num=self.generation_num,
            ret_all_results=self.ret_all_results,
        )
        print(f"Result Mean: {all_res}")
        print(f"Result Standard Deviation: {all_res}")

        # record operators history and best score
        # self.record_evaluation(all_res)
        return all_res

    def evaluate_for_instance(
        self,
        problem: FJSP_Problem,
    ) -> Any:
        """
        评估当前算子组合在特定算例上的性能

        参数:
            problem: 问题实例

        返回:
            评估结果
        """
        # 记录实验配置
        self.logger.log_config(
            {
                "problem": problem.name,
                "instance": problem.batchNo,
                "objectives": problem.obj_list,
                "num_jobs": problem.jobNum,
                "num_machines": problem.proMachineNum,
                "num_operations": problem.operationNum,
                "population_size": 20,
                "max_generations": self.generation_num,
                "num_evaluations": self.n_evals,
                "ref_point": (
                    self.ref_point.tolist() if self.ref_point is not None else None
                ),
            }
        )

        # 记录算子组合配置
        self.logger.log_operator(
            "operator_combination",
            {
                "op_crossover": (
                    "custom" if self.operators["op_crossover"] else "default"
                ),
                "op_mutation": "custom" if self.operators["op_mutation"] else "default",
                "ma_crossover": (
                    "custom" if self.operators["ma_crossover"] else "default"
                ),
                "ma_mutation": "custom" if self.operators["ma_mutation"] else "default",
            },
        )

        # 设置参考点（如果未设置）
        if self.ref_point is None:
            self.ref_point = problem.ref_points

        # 评估
        result = evaluate(
            ref_point=self.ref_point,
            algorithm=self.algorithm,
            problem=problem,
            operators=self.operators,
            logger=self.logger,
            n_evals=self.n_evals,
            generation_num=self.generation_num,
            ret_all_results=self.ret_all_results,
        )

        return result

    def record_evaluation(self, score):
        # 更新评估计数
        self.evaluation_count += 1
        print(f"Total evaluations: {self.evaluation_count}")

        # 记录到本地历史
        copied_operators = {k: deepcopy(v) for k, v in self.operators.items()}
        self.evaluation_history.append((copied_operators, score))

        # 记录到全局存储
        self.global_storage.record_evaluation(self.operators, score)

        # 更新本地最佳结果
        if score > self.best_score:
            self.best_score = score
            self.best_operators = copied_operators
            print(f"New local best combination found! Score: {self.best_score:.4f}")


# 算子模板定义
op_crossover_template = '''

import numpy as np

from typing import Tuple

def op_crossover(parent1: np.ndarray, parent2: np.ndarray, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
    """工序优先级交叉（连续实数向量）。单点切片拼接，返回与父代同长的两个子代。
    约束：
      - parent1/parent2 都被视为 1D 连续实数向量，不是排列；不做查重与补齐。
      - 只做简单切分拼接，并对结果 clip 到 [0,1] 以增强鲁棒性。
      - 任何异常/尺寸不符，直接返回父代拷贝。
    # 只使用 numpy。把染色体当作长度为 n_vars 的一维连续实数向量（取值通常在 [0,1]）。
    # 严禁把它当成“无重复排列”，严禁使用：查重(in / set)、占位(-1)、循环填充等逻辑。
    # 任何异常或维度不符时，必须原样返回父代（保持长度与 dtype）。

    """
    try:
        # 基本形状检查：必须是一维向量
        if parent1.ndim != 1 or parent2.ndim != 1:
            return parent1.copy(), parent2.copy()

        # 若长度与 n_vars 不一致，做“截断/零填充”到 n_vars 长度
        if parent1.size != n_vars or parent2.size != n_vars:
            p1 = np.zeros(n_vars, dtype=(parent1.dtype if parent1.size else float))
            p2 = np.zeros(n_vars, dtype=(parent2.dtype if parent2.size else float))
            m1 = min(n_vars, parent1.size)
            m2 = min(n_vars, parent2.size)
            if m1 > 0: p1[:m1] = parent1[:m1]
            if m2 > 0: p2[:m2] = parent2[:m2]
        else:
            p1, p2 = parent1, parent2

        # 边界情形
        if n_vars <= 1:
            return p1.copy(), p2.copy()

        # 单点交叉：只做切片拼接，不做任何查重/占位/补齐
        cp = np.random.randint(1, n_vars)
        c1 = np.concatenate([p1[:cp], p2[cp:]], axis=0)
        c2 = np.concatenate([p2[:cp], p1[cp:]], axis=0)

        # 加一道保险：clip 到 [0,1]（若你的流程不需要，可删掉）
        c1 = np.clip(c1, 0.0, 1.0)
        c2 = np.clip(c2, 0.0, 1.0)

        return c1, c2
    except Exception:
        # 任意异常：回退父代
        return parent1.copy(), parent2.copy()
'''


def op_crossover(
    parent1: np.ndarray, parent2: np.ndarray, n_vars: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    工序优先级交叉算子

    参数:
        parent1: 父代1的工序优先级染色体，形状为(n_vars,)
        parent2: 父代2的工序优先级染色体，形状为(n_vars,)
        n_vars: 染色体长度

    返回:
        两个子代染色体 (child1, child2)
    """
    # 实现工序优先级交叉
    if n_vars < 2:
        # 极小长度时直接拷贝，避免 randint(1,1) 报错
        return parent1.copy(), parent2.copy()
    cp = np.random.randint(1, n_vars)
    child1 = np.concatenate([parent1[:cp], parent2[cp:]])
    child2 = np.concatenate([parent2[:cp], parent1[cp:]])
    return child1, child2


op_mutation_template = '''
import numpy as np

def op_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    """工序优先级变异算子
    Args:
        solution: 工序优先级染色体 (n_vars,)
        n_vars: 染色体长度
    Returns:
        变异后的染色体 (长度与输入相同)
    # 注意：
    # 1. solution 是一维 numpy.ndarray（长度 = n_vars），不是 list。
    # 2. 禁止使用 pop / append / remove / insert 等列表操作。
    # 3. 必须保证返回值仍是 numpy.ndarray，长度不变。
    # 4. 出错时直接返回输入的拷贝，保证进化不中断。
    """
    try:
        # 确保输入长度正确
        if solution.ndim != 1:
            return solution.copy()
        if solution.size != n_vars:
            # 自动裁剪或零填充
            new_solution = np.zeros(n_vars, dtype=solution.dtype)
            m = min(solution.size, n_vars)
            new_solution[:m] = solution[:m]
        else:
            new_solution = solution.copy()

        if n_vars < 2:
            return new_solution

        # 随机选择两个位置交换
        i, j = np.random.choice(n_vars, 2, replace=False)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        # clip 到 [0,1]（如果是连续编码）
        new_solution = np.clip(new_solution, 0.0, 1.0)

        return new_solution
    except Exception:
        return solution.copy()
'''


def op_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    """
    工序优先级变异算子

    参数:
        solution: 工序优先级染色体，形状为(n_vars,)
        n_vars: 染色体长度

    返回:
        变异后的染色体
    """
    # 实现工序优先级变异
    i, j = np.random.choice(n_vars, 2, replace=False)
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


ma_crossover_template = '''
import numpy as np
from typing import Tuple

def ma_crossover(parent1: np.ndarray, parent2: np.ndarray, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
    """机器分配交叉算子
    Args:
        parent1: 父代1的机器选择染色体 (n_vars,)
        parent2: 父代2的机器选择染色体 (n_vars,)
        n_vars: 染色体长度
    Returns:
        两个子代染色体 (child1, child2)
    """
    # 随机选择交叉点
    cross_point = np.random.randint(1, n_vars)
    
    # 创建子代
    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
    
    return child1, child2
'''


def ma_crossover(
    parent1: np.ndarray, parent2: np.ndarray, n_vars: int
) -> Tuple[np.ndarray, np.ndarray]:
    """机器分配交叉算子
    Args:
        parent1: 父代1的机器选择染色体 (n_vars,)
        parent2: 父代2的机器选择染色体 (n_vars,)
        n_vars: 染色体长度
    Returns:
        两个子代染色体 (child1, child2)
    """
    # 随机选择交叉点
    cross_point = np.random.randint(1, n_vars)

    # 创建子代
    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])

    return child1, child2


ma_mutation_template = '''
import numpy as np
from typing import List

def ma_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    """机器分配变异算子
    Args:
        solution: 机器选择染色体 (n_vars,)
        n_vars: 染色体长度
    Returns:
        变异后的染色体
    """
    # 随机选择一个位置变异
    idx = np.random.randint(0, n_vars)
    new_solution = solution.copy()
    
    # 随机扰动值
    perturbation = np.random.uniform(-0.1, 0.1)
    new_value = np.clip(new_solution[idx] + perturbation, 0, 1)
    new_solution[idx] = new_value
    
    return new_solution
'''


def ma_mutation(solution: np.ndarray, n_vars: int) -> np.ndarray:
    """
    机器分配变异算子

    参数:
        solution: 机器选择染色体，形状为(n_vars,)
        n_vars: 染色体长度

    返回:
        变异后的染色体
    """
    # 实现机器选择变异
    idx = np.random.randint(0, n_vars)
    new_solution = solution.copy()
    perturbation = np.random.uniform(-0.1, 0.1)
    new_value = np.clip(new_solution[idx] + perturbation, 0, 1)
    new_solution[idx] = new_value
    return new_solution


if __name__ == "__main__":
    # 参数初始化
    obj_list = ["makespan", "max_load"]
    generation_num = 100  # GA迭代次数
    n_evals = 2  # 每次评估的重复次数
    pop_size = 200  # 种群大小 对其eoh20
    exp_name = "FJSP_Operator_Evaluation"  # 实验名称

    # 定义基准目录
    benchmark = "brandimarte"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(BASE_DIR, "Instances", benchmark)

    # 获取所有算例文件路径
    instance_paths = glob.glob(os.path.join(instances_dir, "*.txt"))
    instance_paths.sort()

    # 如果没有找到文件，尝试其他扩展名
    if not instance_paths:
        instance_paths = glob.glob(os.path.join(instances_dir, "*"))
        instance_paths = [p for p in instance_paths if os.path.isfile(p)]
        instance_paths.sort()

    print(f"Found {len(instance_paths)} instances in '{benchmark}' benchmark:")
    for i, path in enumerate(instance_paths):
        print(f"  {i+1}. {os.path.basename(path)}")

    # 评估所有算例
    all_results = {}
    for instance_path in instance_paths[:1]:  # only test one instance
        # 获取算例名称
        instance_name = os.path.basename(instance_path)

        print(f"\n===== Evaluating instance: {instance_name} =====")

        # 创建问题实例
        problem = FJSP_Problem(obj_list, instance_path)

        # 设置参考点
        ref_point = problem.ref_points

        # 创建评估器
        evaluator = MyEvaluation(
            algorithm=NSGA_FJSP,
            problem=problem,
            exp_name=exp_name,
            template=None,  # 不需要模板
            ev_operator_name="",  # 不需要特定算子名称
            ref_point=ref_point,
            generation_num=generation_num,
            n_evals=n_evals,
            pop_size=pop_size,
        )

        # # 设置所有算子
        # evaluator.set_operator("op_crossover", op_crossover)
        # evaluator.set_operator("op_mutation", op_mutation)
        # evaluator.set_operator("ma_crossover", ma_crossover)
        # evaluator.set_operator("ma_mutation", ma_mutation)
        operators_functions = {
            "op_crossover": op_crossover,
            "op_mutation": op_mutation,
            "ma_crossover": ma_crossover,
            "ma_mutation": ma_mutation,
        }

        # 评估算子组合
        result = evaluator.evaluate_combination(None)

        # 保存结果
        all_results[instance_name] = result

        print(
            f"===== Completed evaluation for {instance_name}, Avg HV: {result:.4f} ====="
        )

    # 打印所有结果
    print("\n===== All instance evaluation results =====")
    for instance_name, result in all_results.items():
        print(f"{instance_name}: Avg HV = {result:.4f}")

    # 保存综合结果
    results_dir = "experiments"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{exp_name}_summary.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {results_file}")
