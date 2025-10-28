# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from Logger import Logger
from typing import List, Dict, Callable, Optional, Any
import copy


class NSGA_FJSP(ea.MoeaAlgorithm):
    """
    moea_psy_NSGA2_templet : class - 多染色体的多目标进化NSGA-II算法类

    描述:
        采用NSGA-II进行多目标优化，算法详见参考文献[1]。
        该算法类是内置算法类moea_NSGA2_templet的多染色体版本。
        因此里面的种群对象为支持混合编码的多染色体种群类PsyPopulation类的对象。

    参考文献:
        [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective
        genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary
        Computation, 2002, 6(2):0-197.

    """

    def __init__(
        self,
        problem,
        population,
        MAXGEN=None,
        MAXTIME=None,
        MAXEVALS=None,
        MAXSIZE=None,
        logTras=None,
        verbose=None,
        outFunc=None,
        drawing=None,
        dirName=None,
        **kwargs,
    ):
        # 先调用父类构造方法
        super().__init__(
            problem,
            population,
            MAXGEN,
            MAXTIME,
            MAXEVALS,
            MAXSIZE,
            logTras,
            verbose,
            outFunc,
            drawing,
            dirName,
        )
        # save parameters
        self.population = population
        self.problem = problem
        self.ref_point = np.array(problem.ref_points)
        self.kwargs = kwargs
        # population check
        if population.ChromNum == 1:
            raise RuntimeError("传入的种群对象必须是多染色体的种群类型。")
        self.name = "psy-NSGA2"
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序，排序复杂度O(MN^2)
        else:
            self.ndSort = (
                ea.ndsortTNS
            )  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快 O(Nlog(N)~O(N^2))
        self.selFunc = "tour"  # 选择方式，采用锦标赛选择
        # 由于有多个染色体，因此需要用多个重组和变异算子
        self.recOpers = []
        self.mutOpers = []

        # step: 获取所有LLM生成的算子
        self.op_crossover = kwargs.get("op_crossover", None)
        self.op_mutation = kwargs.get("op_mutation", None)
        self.ma_crossover = kwargs.get("ma_crossover", None)
        self.ma_mutation = kwargs.get("ma_mutation", None)
        # step: 为每个染色体配置对应的重组和变异算子
        for i in range(population.ChromNum):
            if i == 0:  # 第一条染色体是工序优先级部分
                # 交叉算子
                if self.op_crossover:
                    recOper = OpCrossoverWrapper(self.problem, self.op_crossover)
                else:
                    recOper = OpCrossover(self.problem)  # 默认工序优先级交叉算子

                # 变异算子
                if self.op_mutation:
                    mutOper = OpMutationWrapper(self.problem, self.op_mutation)
                else:
                    mutOper = OpMutation(self.problem)  # 默认工序优先级变异算子
            elif i == 1:  # 第二条染色体是机器选择部分
                # 交叉算子
                if self.ma_crossover:
                    recOper = MaCrossoverWrapper(self.problem, self.ma_crossover)
                else:
                    recOper = MaCrossover(self.problem)  # 默认机器选择交叉算子

                # 变异算子
                if self.ma_mutation:
                    mutOper = MaMutationWrapper(self.problem, self.ma_mutation)
                else:
                    mutOper = MaMutation(self.problem)  # 默认机器选择变异算子

            # 添加到算子列表
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)

    def reinsertion(self, population, offspring, NUM):
        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            注：这里对原版NSGA-II进行等价的修改：先按帕累托分级和拥挤距离来计算出种群个体的适应度，
            然后调用dup选择算子(详见help(ea.dup))来根据适应度从大到小的顺序选择出个体保留到下一代。
            这跟原版NSGA-II的选择方法所得的结果是完全一样的。
        """
        # 父子两代合并
        population = population + offspring
        # 选择个体保留到下一代
        [levels, criLevel] = self.ndSort(
            population.ObjV, NUM, None, population.CV, self.problem.maxormins
        )  # 对NUM个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV[:, 0] = np.argsort(
            np.lexsort(np.array([dis, -levels])), kind="mergesort"
        )  # 计算适应度
        chooseFlag = ea.selecting(
            "dup", population.FitnV, NUM
        )  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag]

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        self.MAXGEN += 1  # 与moeda区分,确保gen数量一致
        # 初始化日志记录器
        self.logger = None
        if hasattr(self, "experiment_logger"):
            self.logger = self.experiment_logger
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        population.initChrom()  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        # print("=== Initial Population ObjV ===")
        # print(population.ObjV)
        [levels, criLevel] = self.ndSort(
            population.ObjV, NIND, None, population.CV, self.problem.maxormins
        )  # 对NIND个个体进行非支配分层
        population.FitnV = (1 / levels).reshape(
            -1, 1
        )  # 直接根据levels来计算初代个体的适应度

        # ===========================开始进化============================
        run_ok = True  # 本次运行是否全程成功

        while not self.terminated(population):
            try:
                # 选择个体参与进化
                offspring = population[
                    ea.selecting(self.selFunc, population.FitnV, NIND)
                ]

                # 进行进化操作，分别对各个种群染色体矩阵进行重组和变异
                for i in range(population.ChromNum):
                    try:
                        # 重组操作
                        offspring.Chroms[i] = self.recOpers[i].do(offspring.Chroms[i])

                    except Exception as e:
                        print(f"Warning: Recombination operation {i} failed: {e}")
                        print("Keeping original chromosomes for this operation")
                        run_ok = False
                        # 重组失败时保持原染色体不变

                    try:
                        # 变异操作
                        offspring.Chroms[i] = self.mutOpers[i].do(
                            offspring.Encodings[i],  # 编码类型
                            offspring.Chroms[i],  # 染色体数据
                            offspring.Fields[i],  # 字段信息
                        )

                    except Exception as e:
                        print(f"Warning: Mutation operation {i} failed: {e}")
                        print(
                            "Keeping chromosomes after recombination for this operation"
                        )
                        run_ok = False
                        # 变异失败时保持重组后的染色体不变

                # 求进化后个体的目标函数值
                try:
                    self.call_aimFunc(offspring)
                except Exception as e:
                    print(f"Warning: Objective function calculation failed: {e}")
                    print("Using parent fitness values as fallback")
                    run_ok = False
                    # 目标函数计算失败时，使用父代的适应度值（此处按你的实现保持不变）

                # 重插入生成新一代种群
                try:
                    population = self.reinsertion(population, offspring, NIND)
                except Exception as e:
                    print(f"Warning: Reinsertion operation failed: {e}")
                    print("Using previous population as fallback")
                    run_ok = False
                    # 重插入失败时，保持原种群不变

            except Exception as e:
                print(f"Critical error in generation {self.currentGen}: {e}")
                print("Attempting to continue with previous population")
                run_ok = False
                # 发生严重错误时，尝试使用上一代的种群继续进化

            finally:
                self.pop = population  # 把当前种群挂到算法实例，方便 callback 访问
                if hasattr(self, "callback") and callable(self.callback):
                    try:
                        self.callback(self)
                    except Exception as e:
                        print(f"Warning: callback failed at gen {self.currentGen}: {e}")
                # 确保代数计数器递增
                # self.currentGen += 1
        # === 循环结束后补记终止态 ===
        # self.pop = population
        # if hasattr(self, "callback") and callable(self.callback):
        #     try:
        #         self.currentGen += 1
        #         self.callback(self)
        #     except Exception as e:
        #         print(f"Warning: callback at finishing failed: {e}")

        return self.finishing(population), run_ok


# ========================== FJSP专用算子实现 ==========================

# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

# ==============================
# 工具函数
# ==============================


def _machine_uppers_from_problem(problem):
    """
    返回机器段每个位的上界数组 uppers（长度 = operationNum；第 i 位是该工序可用机器数-1）。
    """
    uppers = []
    for op_info in problem.op_mapping:
        ub = max(len(op_info["machines"]) - 1, 0)
        uppers.append(ub)
    return np.array(uppers, dtype=int)


def _rank_normalize_job_segment(seg: np.ndarray) -> np.ndarray:
    """
    对作业内的一段连续优先级做“名次化归一”：保证同一作业内的相对次序而非绝对值。
    seg: shape = (num_ops_of_a_job,)
    """
    if seg.size <= 1:
        return np.array([1.0], dtype=float)
    ranks = np.argsort(np.argsort(seg))  # 0..(n-1)
    return ranks / (seg.size - 1)  # 归一化到 [0,1]


def _adjust_priority_whole_vector(chrom: np.ndarray, job_process) -> np.ndarray:
    """
    对整条“优先级段”按作业内进行名次化归一。chrom shape = (MS_len,)
    """
    out = chrom.copy()
    op_start = 0
    for job in job_process:
        num_ops = len(job["operations"])
        seg = out[op_start : op_start + num_ops]
        out[op_start : op_start + num_ops] = _rank_normalize_job_segment(seg)
        op_start += num_ops
    return out


# ==============================
# 1) 工序优先级：交叉（连续段）
# ==============================


class OpCrossover(ea.Recombination):
    """
    工序优先级交叉（连续段）：单点交叉 + 作业内名次化归一
    假定 OldChrom 的 shape = (Nind, MS_len)，值域在 [0,1] 左右（变异后再 clip）
    """

    def __init__(self, problem, XOVR=0.7):
        self.problem = problem
        self.XOVR = XOVR

    def do(self, OldChrom):
        Nind, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()

        pairs = np.random.permutation(Nind)
        if Nind % 2 == 1:  # 奇数个体，丢弃最后一个或单独保留
            pairs = pairs[:-1]
        pairs = pairs.reshape(-1, 2)

        for p1, p2 in pairs:
            parent1 = NewChrom[p1].copy()
            parent2 = NewChrom[p2].copy()

            if np.random.rand() < self.XOVR:
                cp = np.random.randint(1, Lind)
                child1 = np.concatenate([parent1[:cp], parent2[cp:]])
                child2 = np.concatenate([parent2[:cp], parent1[cp:]])

                # 合法化：clip 到 [0,1] + 作业内名次化
                child1 = np.clip(child1, 0.0, 1.0)
                child2 = np.clip(child2, 0.0, 1.0)
                child1 = _adjust_priority_whole_vector(child1, self.problem.jobProcess)
                child2 = _adjust_priority_whole_vector(child2, self.problem.jobProcess)
            else:
                child1, child2 = parent1, parent2

            NewChrom[p1] = child1
            NewChrom[p2] = child2

        return NewChrom


# ==============================
# 2) 工序优先级：变异（连续段）
# ==============================


class OpMutation(ea.Mutation):
    """
    工序优先级变异（连续段）：每基因小扰动 + 作业内名次化归一。
    """

    def __init__(self, problem, Pm=0.1, step=0.1):
        self.problem = problem
        self.Pm = Pm
        self.step = step

    def do(self, Encoding, OldChrom, FieldDR):
        Nind, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()

        for i in range(Nind):
            vec = NewChrom[i].copy()
            # 基因级变异
            mask = np.random.rand(Lind) < self.Pm
            noise = np.random.uniform(-self.step, self.step, size=Lind)
            vec[mask] = np.clip(vec[mask] + noise[mask], 0.0, 1.0)

            # 作业内名次化，以保持相对顺序
            vec = _adjust_priority_whole_vector(vec, self.problem.jobProcess)
            NewChrom[i] = vec

        return NewChrom


# ==============================
# 3) 机器索引：交叉（整数段）
# ==============================


class MaCrossover(ea.Recombination):
    """
    机器索引交叉（整数段）：单点交叉 + 逐位夹紧到 [0, ub_i]；最后取整。
    假定 OldChrom 的 shape = (Nind, machine_len) 且为“整数向量”（dtype 可为浮点，但会被 round）。
    """

    def __init__(self, problem, XOVR=0.7):
        self.problem = problem
        self.XOVR = XOVR
        self.uppers = _machine_uppers_from_problem(problem)  # 每个位的上界

    def _legalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.rint(arr).astype(int)  # 四舍五入转整数
        arr = np.minimum(arr, self.uppers)
        arr = np.maximum(arr, 0)
        return arr

    def do(self, OldChrom):
        Nind, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()

        pairs = np.random.permutation(Nind)
        if Nind % 2 == 1:
            pairs = pairs[:-1]
        pairs = pairs.reshape(-1, 2)

        for p1, p2 in pairs:
            parent1 = NewChrom[p1].copy()
            parent2 = NewChrom[p2].copy()

            if np.random.rand() < self.XOVR:
                cp = np.random.randint(1, Lind)
                child1 = np.concatenate([parent1[:cp], parent2[cp:]])
                child2 = np.concatenate([parent2[:cp], parent1[cp:]])
                # 合法化为整数并夹紧
                child1 = self._legalize(child1)
                child2 = self._legalize(child2)
            else:
                child1, child2 = self._legalize(parent1), self._legalize(parent2)

            NewChrom[p1] = child1
            NewChrom[p2] = child2

        return NewChrom


# ==============================
# 4) 机器索引：变异（整数段）
# ==============================


class MaMutation(ea.Mutation):
    """
    机器索引变异（整数段）：
      - 70% 概率：对某位做邻域 ±1（并夹紧到 [0, ub_i]）
      - 30% 概率：该位随机重采样 [0, ub_i]
    """

    def __init__(self, problem, Pm=0.1):
        self.problem = problem
        self.Pm = Pm
        self.uppers = _machine_uppers_from_problem(problem)

    def do(self, Encoding, OldChrom, FieldDR):
        Nind, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()

        for i in range(Nind):
            vec = np.rint(NewChrom[i]).astype(int)
            for j in range(Lind):
                if np.random.rand() < self.Pm:
                    ub = int(self.uppers[j])
                    if ub <= 0:
                        vec[j] = 0
                        continue
                    if np.random.rand() < 0.7:
                        # 邻域 ±1
                        step = np.random.choice([-1, 1])
                        vec[j] = int(np.clip(vec[j] + step, 0, ub))
                    else:
                        # 随机重采样
                        vec[j] = int(np.random.randint(0, ub + 1))
            NewChrom[i] = vec

        return NewChrom


# ==============================
# 5) LLM 包装器（确保强制合法化）
# ==============================


class OpCrossoverWrapper(ea.Recombination):
    """工序优先级交叉包装：外部函数 -> clip 到 [0,1] -> 作业内名次化"""

    def __init__(self, problem, crossover_: callable):
        super().__init__()
        self.problem = problem
        self.crossover = crossover_

    def do(self, OldChrom):
        N, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()

        pairs = np.random.permutation(N)
        if N % 2 == 1:
            pairs = pairs[:-1]
        pairs = pairs.reshape(-1, 2)

        for p1, p2 in pairs:
            a = NewChrom[p1].copy()
            b = NewChrom[p2].copy()
            try:
                c1, c2 = self.crossover(a, b, Lind)
            except Exception as e:
                c1, c2 = a, b
                break
            c1 = np.clip(c1, 0.0, 1.0)
            c2 = np.clip(c2, 0.0, 1.0)
            c1 = _adjust_priority_whole_vector(c1, self.problem.jobProcess)
            c2 = _adjust_priority_whole_vector(c2, self.problem.jobProcess)
            NewChrom[p1], NewChrom[p2] = c1, c2

        return NewChrom


class OpMutationWrapper(ea.Mutation):
    """工序优先级变异包装：外部函数 -> clip 到 [0,1] -> 作业内名次化"""

    def __init__(self, problem, mutation_: callable):
        super().__init__()
        self.problem = problem
        self.mutation = mutation_

    def do(self, Encoding, OldChrom, FieldDR):
        N, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()
        for i in range(N):
            try:
                v = self.mutation(NewChrom[i].copy(), Lind)
            except Exception:
                v = NewChrom[i].copy()
            v = np.clip(v, 0.0, 1.0)
            v = _adjust_priority_whole_vector(v, self.problem.jobProcess)
            NewChrom[i] = v
        return NewChrom


class MaCrossoverWrapper(ea.Recombination):
    """机器索引交叉包装：外部函数 -> 四舍五入为整数 -> 逐位夹紧到 [0, ub_i]"""

    def __init__(self, problem, crossover_: callable):
        super().__init__()
        self.problem = problem
        self.crossover = crossover_
        self.uppers = _machine_uppers_from_problem(problem)

    def _legalize(self, arr):
        arr = np.rint(arr).astype(int)
        arr = np.minimum(arr, self.uppers)
        arr = np.maximum(arr, 0)
        return arr

    def do(self, OldChrom):
        N, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()
        pairs = np.random.permutation(N)
        if N % 2 == 1:
            pairs = pairs[:-1]
        pairs = pairs.reshape(-1, 2)
        for p1, p2 in pairs:
            a = NewChrom[p1].copy()
            b = NewChrom[p2].copy()
            try:
                c1, c2 = self.crossover(a, b, Lind)
            except Exception:
                c1, c2 = a, b
            NewChrom[p1] = self._legalize(c1)
            NewChrom[p2] = self._legalize(c2)
        return NewChrom


class MaMutationWrapper(ea.Mutation):
    """机器索引变异包装：外部函数 -> 四舍五入为整数 -> 逐位夹紧到 [0, ub_i]"""

    def __init__(self, problem, mutation_: callable):
        super().__init__()
        self.problem = problem
        self.mutation = mutation_
        self.uppers = _machine_uppers_from_problem(problem)

    def _legalize(self, arr):
        arr = np.rint(arr).astype(int)
        arr = np.minimum(arr, self.uppers)
        arr = np.maximum(arr, 0)
        return arr

    def do(self, Encoding, OldChrom, FieldDR):
        N, Lind = OldChrom.shape
        NewChrom = OldChrom.copy()
        for i in range(N):
            try:
                v = self.mutation(NewChrom[i].copy(), Lind)
            except Exception:
                v = NewChrom[i].copy()
            NewChrom[i] = self._legalize(v)
        return NewChrom
