import numpy as np
import random
from copy import deepcopy
import geatpy as ea  # MOEA/D 模板
import itertools
from math import comb
import time

# 假设 OpCrossover、OpMutation、MaCrossover、MaMutation 及其 Wrapper 已定义


class MOEAD_FJSP(ea.MoeaAlgorithm):
    """
    MOEA/D for Flexible Job Shop Scheduling Problem (FJSP)
    完全兼容 NSGA_FJSP 接口与回调逻辑
    """

    def __init__(
        self,
        problem,
        population,
        T=20,
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

        # 保存参数
        self.population = population
        self.ref_point = np.array(problem.ref_points)
        self.problem = problem
        self.T = T
        self.Pop_size = population.sizes
        # self.gene_size = MAXGEN if MAXGEN else 200

        # 初始化参考点和权重向量
        self._z = np.array(problem.ref_points, dtype=float)
        self._lambda = np.array(
            self._uniform_sampling_vgm(self.Pop_size, len(problem.obj_list)),
            dtype=float,
        )

        # 初始化邻居矩阵
        self.B = self._init_neighbors()

        # 初始化算子列表
        self.recOpers, self.mutOpers = [], []
        self._init_operators(kwargs)

        # currentGen 用于 callback
        self.currentGen = 1

        # 确保日志和计时初始化，避免 terminated() 报错
        self.timeSlot = getattr(self, "timeSlot", 0.0)
        self.passTime = getattr(self, "passTime", 0.0)
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序，排序复杂度O(MN^2)
        else:
            self.ndSort = (
                ea.ndsortTNS
            )  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快 O(Nlog(N)~O(N^2))

    def _uniform_sampling_vgm(self, pop, obj):
        """
        Das and Dennis 单纯形格子设计生成权重向量
        """
        n, m = pop, obj
        H = 1
        while comb(H + m - 1, m - 1) < n:
            H += 1

        combinations = list(itertools.combinations(range(1, H + m), m - 1))
        weight_vectors = []
        for comb_tuple in combinations:
            points = [0] + list(comb_tuple) + [H + m]
            weights = [points[i] - points[i - 1] - 1 for i in range(1, len(points))]
            total = sum(weights)
            weight_vectors.append([w / total for w in weights])
        return weight_vectors

    def _init_neighbors(self):
        # 确保 _lambda 是 numpy array
        lamb = np.array(self._lambda, dtype=float)
        dist_matrix = np.linalg.norm(lamb[:, None, :] - lamb[None, :, :], axis=2)
        B = [
            np.argsort(dist_matrix[i])[: self.T].tolist() for i in range(self.Pop_size)
        ]
        return B

    def _init_operators(self, kwargs):
        op_crossover = kwargs.get("op_crossover", None)
        op_mutation = kwargs.get("op_mutation", None)
        ma_crossover = kwargs.get("ma_crossover", None)
        ma_mutation = kwargs.get("ma_mutation", None)
        for i in range(self.population.ChromNum):
            if i == 0:
                recOper = (
                    OpCrossoverWrapper(self.problem, op_crossover)
                    if op_crossover
                    else OpCrossover(self.problem)
                )
                mutOper = (
                    OpMutationWrapper(self.problem, op_mutation)
                    if op_mutation
                    else OpMutation(self.problem)
                )
            else:
                recOper = (
                    MaCrossoverWrapper(self.problem, ma_crossover)
                    if ma_crossover
                    else MaCrossover(self.problem)
                )
                mutOper = (
                    MaMutationWrapper(self.problem, ma_mutation)
                    if ma_mutation
                    else MaMutation(self.problem)
                )
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)

    def Tchebycheff(self, individual, z, weights):
        arr = weights * np.abs(individual.ObjV - z)
        return np.max(arr)  # 返回一个标量

    def reinsertion(self, population, offspring, NUM):
        """
        基于邻域和目标分解的Tchebycheff距离选择个体
        population 和 offspring 合并后，按 Tchebycheff 距离更新
        """
        # 合并父代和子代
        population = population + offspring

        # 初始化新的种群
        new_population = []

        for i in range(self.Pop_size):
            # 当前个体的目标值
            individual = population[i]

            # 对该个体，计算其与邻域中所有个体的Tchebycheff距离
            best_individual = None
            best_distance = float("inf")

            # 计算 Tchebycheff 距离，选择与邻域更适应的个体
            for bi in self.B[i]:
                # 计算当前个体与邻居个体的 Tchebycheff 距离
                f_old = self.Tchebycheff(population[bi], self._z, self._lambda[bi])
                f_new = self.Tchebycheff(individual, self._z, self._lambda[bi])

                # 如果新个体的 Tchebycheff 距离更小，更新该邻域的个体
                if f_new < f_old:
                    if f_new < best_distance:
                        best_distance = f_new
                        best_individual = individual

            # 如果有更优的个体，更新
            if best_individual is not None:
                new_population.append(best_individual)
            else:
                # 如果没有更新，则保留当前个体
                new_population.append(individual)

        # 将更新后的种群返回
        return new_population

    def run(self, prophetPop=None):
        population = self.population
        NIND = population.sizes
        population.initChrom()
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]
        self.call_aimFunc(population)
        EP = []

        run_ok = True
        for gen in range(self.MAXGEN):
            try:
                offspring = population[ea.selecting("tour", population.FitnV, NIND)]
                for i in range(self.Pop_size):
                    j, k = random.randint(0, self.T - 1), random.randint(0, self.T - 1)
                    child = deepcopy(population[self.B[i][j]])
                    for c in range(self.population.ChromNum):
                        try:
                            child.Chroms[c] = self.recOpers[c].do(child.Chroms[c])
                        except Exception as e:
                            print(f"Warning: recomb {c} failed: {e}")
                        try:
                            child.Chroms[c] = self.mutOpers[c].do(
                                child.Encodings[c], child.Chroms[c], child.Fields[c]
                            )
                        except Exception as e:
                            print(f"Warning: mutation {c} failed: {e}")
                    try:
                        self.call_aimFunc(child)
                    except Exception as e:
                        print(f"Warning: objective func failed: {e}")

                    self._z = np.minimum(self._z, child.ObjV)

                    for bi in self.B[i]:
                        f_old = self.Tchebycheff(
                            population[bi], self._z, self._lambda[bi]
                        )
                        f_new = self.Tchebycheff(child, self._z, self._lambda[bi])
                        if f_new < f_old:
                            population[bi] = deepcopy(child)

                    _remove, dominateY = [], False
                    for e in EP:
                        if self.Dominate(child, e):
                            _remove.append(e)
                        elif self.Dominate(e, child):
                            dominateY = True
                            break
                    if not dominateY:
                        EP.append(deepcopy(child))
                        for r in _remove:
                            EP.remove(r)

                self.pop = population
                if hasattr(self, "callback") and callable(self.callback):
                    try:
                        self.callback(self)
                    except Exception as e:
                        print(f"Warning: callback failed at gen {self.currentGen}: {e}")

                self.currentGen += 1

            except Exception as e:
                print(f"Critical error in generation {self.currentGen}: {e}")
                run_ok = False

        return EP, run_ok

    def Dominate(self, Pop1, Pop2):
        """
        判断 Pop1 是否支配 Pop2
        适用于最小化目标问题
        """
        obj1 = Pop1.ObjV  # 目标向量
        obj2 = Pop2.ObjV

        # Pop1 支配 Pop2 的条件：所有目标 <=，且至少一个 <
        if np.all(obj1 <= obj2) and np.any(obj1 < obj2):
            return True
        return False


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
