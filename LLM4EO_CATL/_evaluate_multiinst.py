import os
import glob
import json
import numpy as np
from typing import List, Dict, Callable, Optional, Any, Union, Tuple
from Problem.FJSP.FJSP_Problem import FJSP_Problem

# from Algrithm.MOEDA import MOEAD_FJSP
from Algrithm.NSGA3 import NSGA_FJSP

# from _storage import GlobalResultStorage
from Logger import Logger
import random
import geatpy as ea
import threading
from copy import deepcopy
import inspect
import time

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
    save_iteration_data: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行优化算法并返回完整结果（添加迭代记录功能）

    新增参数:
        save_iteration_data: 是否保存迭代数据
        save_dir: 数据保存目录
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

    # 创建迭代数据记录器
    iteration_records = []

    # 自定义回调函数用于记录迭代数据
    def callback(alg):
        gen = int(alg.currentGen)
        pop = getattr(alg, "pop", None) or getattr(alg, "population", None)
        if pop is None or pop.ObjV is None:
            return

        objv = pop.ObjV  # (N, M)
        N, M = objv.shape

        # 本代第一前沿
        levels, _ = alg.ndSort(objv, N, None, pop.CV, alg.problem.maxormins)
        pareto_mask = levels == 1
        pareto_front = objv[pareto_mask].tolist()

        # 人群目标统计
        stats = {
            "min": np.min(objv, axis=0).tolist(),
            "max": np.max(objv, axis=0).tolist(),
            "mean": np.mean(objv, axis=0).tolist(),
        }

        # HV（可选）
        hv_val = None
        if hasattr(alg, "ref_point") and alg.problem.M > 1:
            try:
                hv_val = ea.indicator.HV(objv, PF=np.array(alg.ref_point))
            except Exception:
                hv_val = None

        record = {
            "generation": gen,
            "timestamp": time.time(),
            "population_size": int(getattr(pop, "sizes", getattr(pop, "size", N))),
            "pareto_front": pareto_front,
            "pop_stats": stats,
            "hv": hv_val,
        }
        iteration_records.append(record)

        # —— 所有代统一写在一个文件 —— #
        if save_iteration_data and save_dir:
            out_path = os.path.join(save_dir, "iterations.json")
            os.makedirs(save_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(iteration_records, f, indent=2)

    # 设置回调函数
    myAlgorithm.callback = callback

    # 运行算法
    # BestIndi, resultOr = myAlgorithm.run()
    # 获取帕累托前沿
    # pareto_front = np.array([ind.ObjV for ind in BestIndi], dtype=float).reshape(-1, 2)
    [BestIndi, population], resultOr = myAlgorithm.run()

    # 获取帕累托前沿
    pareto_front = BestIndi.ObjV
    pareto_front = BestIndi.ObjV  # NSGA

    # 计算HV指标
    if resultOr:
        hv = ea.indicator.HV(pareto_front, PF=np.array(ref_point))
    else:
        hv = 0

    # 返回完整结果
    return resulst_calculation(
        pareto_front, hv, generation_num, pop_size, iteration_records
    )


# 计算指标（添加迭代数据）
def resulst_calculation(
    pareto_front, hv, generation_num, pop_size, iteration_records=None
):
    num_solutions = len(pareto_front)
    avg_makespan = np.mean(pareto_front[:, 0])
    avg_utilization = np.mean(pareto_front[:, 1])
    min_makespan = np.min(pareto_front[:, 0])
    max_utilization = np.max(pareto_front[:, 1])

    return {
        "hv": hv,
        "pareto_front": pareto_front.tolist(),
        "makespan": min_makespan,
        "utilization": max_utilization,
        "generations": generation_num,
        "population_size": pop_size,
        "num_solutions": num_solutions,
        "avg_makespan": avg_makespan,
        "avg_utilization": avg_utilization,
        "min_makespan": min_makespan,
        "max_utilization": max_utilization,
        "iteration_records": iteration_records,  # 添加迭代记录
    }


def evaluate(
    algorithm: Callable,
    problems: Dict[str, FJSP_Problem],
    operators: Dict[str, Callable],
    logger: Logger,
    n_evals: int = 11,
    generation_num: int = 20,
    ret_all_results: bool = False,
    pop_size: int = 200,
    save_dir: Optional[str] = None,
    save_iteration_data: bool = False,
) -> Any:
    """
    多次评估算子组合的性能（添加保存路径和迭代记录）

    新增参数:
        save_dir: 结果保存目录
        save_iteration_data: 是否保存迭代数据
    """
    all_res = []
    for instance_name, problem in problems.items():
        if save_dir:
            instance_dir = os.path.join(save_dir, instance_name) if save_dir else None
        print(f"===================instances_{instance_name}")

        for i in range(n_evals):
            print(f"===================n_evals_{i}")
            # 设置随机种子
            np.random.seed(i)
            random.seed(i)

            # 创建运行目录
            if save_dir:
                run_dir = (
                    os.path.join(instance_dir, f"run_{i}") if instance_dir else None
                )

            # 执行优化
            res = main_op(
                ref_point=problem.ref_points,
                algorithm=algorithm,
                problem=problem,
                generation_num=generation_num,
                pop_size=pop_size,
                operators=operators,
                save_iteration_data=save_iteration_data,
            )
            all_res.append(res)

            # 保存每次运行的结果
            if save_dir:
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "result.json"), "w") as f:
                    json.dump(res, f, indent=2)

    # 计算平均HV
    avg_hv = np.mean([r["hv"] for r in all_res])
    print(f"===================avg_hv_{avg_hv}")

    # 校验hv
    if np.isnan(avg_hv) or avg_hv < 0:
        raise ValueError(f"Invalid HV value: {avg_hv}")

    # 保存汇总结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        summary = {"avg_hv": avg_hv, "results": all_res}
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if ret_all_results:
        return all_res
    else:
        return avg_hv


class MultiEvaluation(Evaluation):
    def __init__(
        self,
        algorithm,
        instance_paths,
        exp_name,
        template,
        ev_operator_name,
        obj_list,
        generation_num=20,
        n_evals=6,
        pop_size=200,
        ret_all_results=False,
        output_dir="experiments",
        save_iteration_data=False,
    ):
        """
        初始化评估器（添加输出目录和迭代记录选项）

        新增参数:
            output_dir: 结果输出目录
            save_iteration_data: 是否保存迭代数据
        """
        super().__init__(template_program=template, timeout_seconds=1500)
        self.n_evals = n_evals
        self.ret_all_results = ret_all_results
        self.algorithm = algorithm
        self.generation_num = generation_num
        self.exp_name = exp_name
        self.pop_size = pop_size
        self.ev_operator_name = ev_operator_name
        self.operators = {
            "op_crossover": None,
            "op_mutation": None,
            "ma_crossover": None,
            "ma_mutation": None,
        }
        self.obj_list = obj_list
        self.output_dir = output_dir
        self.save_iteration_data = save_iteration_data

        # 本地历史记录和最佳结果追踪
        self.evaluation_history = []
        self.best_score = float("-inf")
        self.best_operators = {}
        self.evaluation_count = 0

        # 初始化问题实例
        self.ini_problems(instance_paths)

    def ini_problems(self, instance_paths):
        self.train_paths = []
        self.train_problems = {}
        self.test_paths = []
        self.test_problems = {}

        # 计算训练集比例
        train_instance_rate = 1  # 测试全部的结果

        for i, path in enumerate(instance_paths):
            instance_name = os.path.basename(path)

            # 训练问题生成
            if i <= round(len(instance_paths) * train_instance_rate):
                self.train_paths.append(instance_name)
                self.train_problems[instance_name] = FJSP_Problem(self.obj_list, path)
            else:
                self.test_paths.append(path)
                self.test_problems[instance_name] = FJSP_Problem(self.obj_list, path)

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
                ):  # ev_operator_name is used llm operatorsd
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

        print("waitting for evaluate...")
        # 算例初始化
        eval_problems = self.train_problems
        dataset_type = "train"

        # 调用evaluate计算hv
        all_res = evaluate(
            algorithm=self.algorithm,
            problems=eval_problems,
            operators=self.operators,
            logger=None,
            n_evals=self.n_evals,
            generation_num=self.generation_num,
            ret_all_results=True,
            pop_size=self.pop_size,
        )

        # self.record_evaluation(res)
        # 计算平均HV
        avg_hv = np.mean([r["hv"] for r in all_res])
        print(f"Average HV for {dataset_type} set: {avg_hv:.4f}")

        # 记录评估历史
        self.record_evaluation(avg_hv)
        return avg_hv

    def evaluate_combination(
        self,
        use_train_set: bool = True,
        operators: Optional[Dict[str, Union[Callable, str]]] = None,
    ) -> float:
        """
        评估特定算子组合的性能（添加训练/测试集选择）

        参数:
            use_train_set: 是否使用训练集（True）或测试集（False）
            operators: 算子字典
        """
        # 设置算子
        if operators:
            for op_name, op_func in operators.items():
                if isinstance(op_func, str):
                    callable_func = string_to_callable(op_func, op_name)
                    if callable_func:
                        self.set_operator(op_name, callable_func)
                    else:
                        print(f"Warning: Failed to compile {op_name} from source code")
                else:
                    self.set_operator(op_name, op_func)

        # 选择数据集
        if use_train_set:
            eval_problems = self.train_problems
            dataset_type = "train"
        else:
            eval_problems = self.test_problems
            dataset_type = "test"

        # 创建保存目录
        save_dir = os.path.join(
            self.output_dir, self.exp_name, dataset_type, time.strftime("%Y%m%d-%H%M%S")
        )

        # 调用evaluate计算hv
        all_res = evaluate(
            algorithm=self.algorithm,
            problems=eval_problems,
            operators=self.operators,
            logger=None,
            n_evals=self.n_evals,
            generation_num=self.generation_num,
            ret_all_results=True,
            pop_size=self.pop_size,
            save_dir=save_dir,
            save_iteration_data=self.save_iteration_data,
        )

        # 计算平均HV
        avg_hv = np.mean([r["hv"] for r in all_res])
        print(f"Average HV for {dataset_type} set: {avg_hv:.4f}")

        # 记录评估历史
        self.record_evaluation(avg_hv)
        return avg_hv

    def record_evaluation(self, score):
        # 更新评估计数
        self.evaluation_count += 1
        print(f"Total evaluations: {self.evaluation_count}")

        # 记录到本地历史
        copied_operators = {k: deepcopy(v) for k, v in self.operators.items()}
        self.evaluation_history.append((copied_operators, score))

        # 更新本地最佳结果
        if score > self.best_score:
            self.best_score = score
            self.best_operators = copied_operators
            print(f"New local best combination found! Score: {self.best_score:.4f}")


if __name__ == "__main__":
    # 参数初始化
    obj_list = ["makespan", "total_load"]  # "max_load",
    generation_num = 30
    n_evals = 5
    pop_size = 100
    exp_name = "NSGA2"
    output_dir = "experiments"
    save_iteration_data = True  # 是否保存迭代数据

    # 算例比例设置
    # train_instance_rate = 0.3

    # 定义基准目录
    benchmark = "brandimarte"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(BASE_DIR, "Instances", benchmark)

    # 获取所有算例文件路径
    instance_paths = glob.glob(os.path.join(instances_dir, "*.txt"))
    if not instance_paths:
        instance_paths = glob.glob(os.path.join(instances_dir, "*"))
        instance_paths = [p for p in instance_paths if os.path.isfile(p)]
    instance_paths.sort()

    print(f"Found {len(instance_paths)} instances in '{benchmark}' benchmark")

    # 确定算子
    default_operators_source = {
        "op_crossover": inspect.getsource(op_crossover),
        "op_mutation": inspect.getsource(op_mutation),
        "ma_crossover": inspect.getsource(ma_crossover),
        "ma_mutation": inspect.getsource(ma_mutation),
    }

    # 创建评估器
    evaluator = MultiEvaluation(
        # algorithm=MOEAD_FJSP,
        algorithm=NSGA_FJSP,
        instance_paths=instance_paths,
        exp_name=exp_name,
        template=None,
        ev_operator_name="",
        generation_num=generation_num,
        n_evals=n_evals,
        pop_size=pop_size,
        output_dir=output_dir,
        save_iteration_data=save_iteration_data,
    )

    evaluator.save_operatorstr_bydict(default_operators_source)

    # 在训练集上评估
    print("\n===== Evaluating on TRAIN set =====")
    train_result = evaluator.evaluate_combination(use_train_set=True)
    print(f"Train set average HV: {train_result:.4f}")

    # 在测试集上评估
    # print("\n===== Evaluating on TEST set =====")
    # test_result = evaluator.evaluate_combination(use_train_set=False)
    # print(f"Test set average HV: {test_result:.4f}")

    # 打印最终结果
    print("\n===== Final Results =====")
    print(f"Train HV: {train_result:.4f}")
    # print(f"Test HV: {test_result:.4f}")
