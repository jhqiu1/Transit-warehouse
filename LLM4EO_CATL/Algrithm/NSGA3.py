# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from Logger import Logger
from typing import List, Dict, Callable, Optional, Any
import copy
import numpy as np
import itertools
from math import comb


class NSGA3_FJSP(ea.MoeaAlgorithm):
    """

    描述:
        引入了一组 参考点（reference directions），把个体与参考点关联起来，通过 小生境 (niche) 分配来保证在高维目标空间里的分布均匀性。

    参考文献:
        [1] Deb K., Jain H. "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints." IEEE TEVC, 2014.

    """

    def __init__(
        self,
        problem,
        population,
        MAXGEN=None,
        MAXTIME=None,
        MAXEVALS=None,
        MAXSIZE=None,
        logTras=1,
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
        self.name = "psy-NSGA3"
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
        self.ref_dirs = self.crtRefDirs(self.problem.M, nPoints=100)  # 可调点数

    def crtRefDirs(self, M, nPoints):
        """
        简单实用版 NSGA-III 参考方向 (Das & Dennis simplex-lattice)
        返回: ndarray, shape = (K, M), 每行和为 1
        """
        seed = 1
        if M < 2:
            raise ValueError("M must be >= 2")
        if nPoints < M:
            # 至少能覆盖每个轴向方向
            nPoints = M

        rng = np.random.default_rng(seed)

        # 1) 选最小 H, 使 C(H+M-1, M-1) >= nPoints
        H = 1
        while comb(H + M - 1, M - 1) < nPoints:
            H += 1

        # 2) 生成单纯形格点（stars & bars）
        #    在 [0, H+M-2] 里选 M-1 个切点，段长-1 即每维整数份数，和为 H
        dirs = []
        for cuts in itertools.combinations(range(H + M - 1), M - 1):
            points = (-1,) + cuts + (H + M - 1,)
            counts = [points[i + 1] - points[i] - 1 for i in range(M)]  # 每维整数份
            dirs.append([c / H for c in counts])

        dirs = np.asarray(dirs, dtype=float)

        # 3) 确保轴向单位向量在集合中（有时 H 太小会没有正好 (1,0,...,0) 这样的格点）
        unit_axes = np.eye(M, dtype=float)

        # 用近似相等判断避免浮点误差
        def _row_in(a, v, tol=1e-12):
            return np.any(np.all(np.abs(a - v) <= tol, axis=1))

        extras = [u for u in unit_axes if not _row_in(dirs, u)]
        if extras:
            dirs = np.vstack([dirs, np.asarray(extras)])

        # 4) 如数量过多，简单抽样到 nPoints（先保留单位向量，再随机采样其余）
        if dirs.shape[0] > nPoints:
            keep = []
            # 保留单位向量（若存在于 dirs）
            for u in unit_axes:
                idx = np.where(np.all(np.abs(dirs - u) <= 1e-12, axis=1))[0]
                if idx.size > 0:
                    keep.append(idx[0])
            keep = list(dict.fromkeys(keep))  # 去重并保持顺序

            # 其余可选索引
            all_idx = np.arange(dirs.shape[0])
            rest_idx = np.setdiff1d(
                all_idx, np.array(keep, dtype=int), assume_unique=False
            )

            need = max(0, nPoints - len(keep))
            if need > 0:
                sampled = rng.choice(rest_idx, size=need, replace=False)
                sel = np.concatenate([np.array(keep, dtype=int), sampled])
            else:
                sel = np.array(keep[:nPoints], dtype=int)

            dirs = dirs[sel]

        # 5) 数值清理：截断 & 归一，保证每行和为 1
        dirs = np.clip(dirs, 0.0, None)
        sums = dirs.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        dirs = dirs / sums
        return dirs

    def reinsertion(self, population, offspring, NUM):
        """
        NSGA-III 环境选择：使用参考点进行小生境分配。
        """
        # 父子合并
        population = population + offspring

        # 非支配排序
        [levels, criLevel] = ea.ndsortTNS(
            population.ObjV, NUM, None, population.CV, self.problem.maxormins
        )

        # 如果刚好分层等于 NUM，就直接保留
        if len(np.where(levels <= criLevel)[0]) == NUM:
            chooseFlag = levels <= criLevel
            return population[chooseFlag]

        # 否则，需要在最后一层做小生境选择
        lastIdx = np.where(levels == criLevel)[0]  # 最后一层索引
        chosen = np.where(levels < criLevel)[0].tolist()  # 已确定保留
        K = NUM - len(chosen)  # 还需要补充的个体数

        # ===== NSGA-III reference point association =====
        # 归一化目标值
        F = population.ObjV
        fmin = F.min(axis=0)
        fmax = F.max(axis=0)
        normF = (F - fmin) / (fmax - fmin + 1e-12)

        # 计算每个个体到参考点的垂直距离
        def perpendicular_distance(point, ref_dir):
            ref_dir = ref_dir / (np.linalg.norm(ref_dir) + 1e-12)
            projection = np.dot(point, ref_dir)
            vec = point - projection * ref_dir
            return np.linalg.norm(vec)

        # 建立参考点到最后一层个体的映射
        ref_to_inds = {i: [] for i in range(len(self.ref_dirs))}
        for i in lastIdx:
            dists = np.array(
                [perpendicular_distance(normF[i], r) for r in self.ref_dirs]
            )
            ref_idx = np.argmin(dists)  # 归属到最近的参考点
            ref_to_inds[ref_idx].append(i)

        # 小生境选择：轮流从参考点选择个体
        added = []
        while len(added) < K:
            for r, inds in ref_to_inds.items():
                if inds:
                    chosen_idx = inds.pop(np.random.randint(len(inds)))
                    added.append(chosen_idx)
                    if len(added) == K:
                        break

        chosen += added
        return population[chosen]

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
        # === 循环结束后补记终止态 ===
        self.pop = population
        if hasattr(self, "callback") and callable(self.callback):
            try:
                self.callback(self)
            except Exception as e:
                print(f"Warning: callback at finishing failed: {e}")

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
                continue
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
