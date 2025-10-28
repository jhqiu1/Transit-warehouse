import numpy as np
import random
import copy
import geatpy as ea  # 导入geatpy库


class FJSP_Problem(ea.Problem):  # 继承Problem父类
    def __init__(self, obj_list, instance_path):
        name = "FJSP_Problem"  # 初始化name(函数名称,可以随意设置)
        self.obj_list = obj_list
        self.ref_points = []
        self.M = len(obj_list)  # 初始化M(目标维数)
        maxormins = [1] * self.M  # 初始化目标最小最大化标记列表,1:min；-1:max

        self.batchNo = instance_path

        data = self.get_init_data(instance_path)
        self.data = data

        Dims = data["Dims"]  # 初始化Dim（决策变量维数）
        varTypes = data["varTypes"]  # 初始化决策变量类型，0：连续；1：离散
        lb = data["lbs"]  # 决策变量下界
        ub = data["ubs"]  # 决策变量上界
        lbin = [1] * Dims  # 决策变量下边界
        ubin = [1] * Dims  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(
            self, name, self.M, maxormins, Dims, varTypes, lb, ub, lbin, ubin
        )

        self.Dims = Dims
        self.ubs = np.array(ub)
        self.lbs = np.array(lb)
        self.MS_len = data["MS_len"]
        self.gene_ini_pops(pop=100)  # 初始化种群

    def get_init_data(self, instance_path):
        # 读取数据并
        content = self.read_data(instance_path)
        # 获取job信息
        self.jobProcess = self.createJobs(content)
        # 初始化FJSP解码器
        self.FJSP_Decoder_ini()
        # 转成ea能处理的格式
        self.eadata = self.transformEAData()
        print("Creating initial data...")
        return self.eadata
        # 对问题做初始化

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def createJobs(self, content):
        self.xNum = 0
        self.operationNum = 0
        contentList = content.strip().split("\n")
        baseData = contentList[0].split()
        jobNum = int(baseData[0])
        proMachineNum = int(baseData[1])
        self.jobNum = jobNum
        self.proMachineNum = proMachineNum

        operation_code = 0
        jobs = []
        for line in contentList[1:]:
            job_info = {}
            numbers = list(map(int, line.split()))
            job_id = len(jobs)
            job_info["job_id"] = job_id
            job_info["operations"] = []
            num_operations = numbers[0]
            index = 1
            for op_num in range(num_operations):
                operation = {}
                operation["operation_id"] = len(job_info["operations"])
                operation["operation_code"] = operation_code
                operation_code = operation_code + 1
                operation["machines"] = {}
                num_machines = numbers[index]
                index += 1
                for mc_no in range(num_machines):
                    machine_id = numbers[index]
                    process_time = numbers[index + 1]
                    operation["machines"][machine_id] = process_time
                    index += 2
                    self.xNum = self.xNum + 1
                job_info["operations"].append(operation)
                self.operationNum = self.operationNum + 1
            jobs.append(job_info)
        return jobs

    def FJSP_Decoder_ini(self):
        """初始化FJSP解码器所需的数据结构"""
        # 构建操作映射表
        self.op_mapping = []
        op_idx = 0
        for job_id, job in enumerate(self.jobProcess):
            for op_id, op in enumerate(job["operations"]):
                self.op_mapping.append(
                    {"job_id": job_id, "op_id": op_id, "machines": op["machines"]}
                )
                op_idx += 1

        # 确定机器总数
        self.machine_count = (
            max(
                max(op["machines"].keys())
                for job in self.jobProcess
                for op in job["operations"]
            )
            + 1
        )

    def transformEAData(self):
        """
        定义决策变量上下界
        前半段：工序优先级（实数，0-1）
        后半段：机器选择（整数索引，每个操作对应可选机器）
        """
        lbs = []
        ubs = []
        varTypes = []

        # 工序优先级部分：实数 [0,1]
        lbs.extend([0] * self.operationNum)
        ubs.extend([1] * self.operationNum)
        varTypes.extend([0] * self.operationNum)  # 连续变量

        # 机器选择部分：整数索引 [0, num_machines-1]
        for op in self.op_mapping:
            num_machines = len(op["machines"])
            lbs.append(0)
            ubs.append(num_machines - 1)
            varTypes.append(1)  # 整数变量

        eadata = {
            "lbs": lbs,
            "ubs": ubs,
            "Dims": len(lbs),
            "MS_len": self.operationNum,
            "varTypes": varTypes,
        }
        return eadata

    def gene_ini_pops(self, pop=10):
        """
        随机生成初始种群
        """
        population = np.zeros((pop, self.Dims))
        fit_results = []

        for i in range(pop):
            op_start = 0
            # 工序优先级部分：随机生成 [0,1]
            for job in self.jobProcess:
                num_ops = len(job["operations"])
                priorities = np.random.rand(num_ops)
                population[i, op_start : op_start + num_ops] = priorities
                op_start += num_ops

            # 机器选择部分：直接生成整数索引
            for j in range(self.operationNum):
                op_info = self.op_mapping[j]
                num_machines = len(op_info["machines"])
                population[i, self.operationNum + j] = np.random.randint(
                    0, num_machines
                )

            # 计算适应度
            fit_results.append(self.calcFunc(population[i, :]))
            # self.save_gantt(
            #     population[i, :], "debug_alg\\log_20250831_100116\\gantt_run_15.png"
            # )
        self.ini_population = population
        self.ini_fit_results = fit_results
        self.ref_points = self.cal_ref_points(fit_results).tolist()
        return fit_results

    def gene_ini_pops_byrules(self, pop=10):
        # 随机生成初始总群，在决策变量范围内随机构建种群
        population = np.zeros((pop, self.Dims))
        fit_results = []
        # 工序优先级部分：确保同一作业内操作顺序正确
        for i in range(pop):
            op_start = 0
            for job in self.jobProcess:
                num_ops = len(job["operations"])
                # 生成严格降序优先级 (确保操作顺序)
                priorities = np.linspace(1.0, 0.1, num_ops)
                population[i, op_start : op_start + num_ops] = priorities
                op_start += num_ops

            # 机器选择部分：确保分配到有效机器
            for j in range(self.operationNum):
                op_info = self.op_mapping[j]
                num_machines = len(op_info["machines"])
                # 在有效区间内均匀分布
                population[i, self.operationNum + j] = np.random.uniform(0, 1 - 1e-6)
            fit_results.append(self.calcFunc(population[i, :]))

        # 储存初始种群和适应度
        self.ini_population = population
        self.ini_fit_results = fit_results
        self.ref_points = self.cal_ref_points(fit_results).tolist()
        return fit_results

    def calcFunc(self, x):
        op_priority = x[: self.operationNum]
        machine_select = x[self.operationNum :]
        op_sequences = self._get_op_sequences(op_priority)
        machine_assignments = self._assign_machines(machine_select)
        makespan, max_load = self._build_schedule(
            op_sequences, machine_assignments, op_priority
        )
        return makespan, max_load  # 返回最大负荷

    def aimFunc(self, pop):
        """
        计算种群的makespan和设备利用率
        :param pop: 染色体矩阵
        :return: (makespan, utilization)
        """
        # 得到决策变量矩阵
        Vars = pop.Phen
        # pop.cV=[]
        pop.ObjV = np.zeros((len(Vars), self.M))
        for i in range(len(Vars)):
            # pop.ObjV[i] = decode(Vars[i])[-1]
            # 变量返回
            makespan, max_load = self.calcFunc(Vars[i])
            pop.ObjV[i, 0] = makespan
            pop.ObjV[i, 1] = max_load

    def _get_op_sequences(self, op_priority):
        op_sequences = {}
        op_start = 0
        for job_id in range(self.jobNum):
            num_ops = len(self.jobProcess[job_id]["operations"])
            # 确保优先级严格降序
            priorities = np.clip(op_priority[op_start : op_start + num_ops], 0.1, 1.0)
            op_sequences[job_id] = np.argsort(priorities)[::-1].tolist()  # 降序排列
            op_start += num_ops
        return op_sequences

    def _assign_machines(self, machine_select):
        """
        根据整数索引直接选择机器
        """
        assignments = []
        for i, op_info in enumerate(self.op_mapping):
            available_machines = sorted(op_info["machines"].keys())
            machine_idx = int(machine_select[i])
            # machine_idx = min(max(machine_idx, 0), len(available_machines) - 1)

            assignments.append(
                {
                    "machine_id": available_machines[machine_idx],
                    "process_time": op_info["machines"][
                        available_machines[machine_idx]
                    ],
                }
            )
        return assignments

    def _build_schedule(self, op_sequences, machine_assignments, op_priority):
        """
        设备驱动的非抢占式派工：每台机器维护本机就绪队列（只含分配到这台机的、前序已完成的工序）。
        当机器空闲且队列非空时，按优先级（数值越大越先）选择一条立即开工。
        """
        # 每台机器的“可用时间”
        machine_available = {}  # {machine_id: time}
        # 每个作业的“已完工时间”
        job_last_end = {j: 0 for j in range(self.jobNum)}
        # 记录调度结果：{global_op_idx: (start, end, machine_id)}
        op_status = {}

        import heapq

        ready_by_machine = {}  # {m: [heap of candidates]}

        def push_ready(m, gidx):
            pr = float(op_priority[gidx])
            if m not in ready_by_machine:
                ready_by_machine[m] = []
            # tie-breaker 用 gidx，确保稳定
            heapq.heappush(ready_by_machine[m], (-pr, gidx))

        def glb2jl(gidx):
            info = self.op_mapping[gidx]
            return info["job_id"], info["op_id"]

        # 初始化：把每个作业的第0道工序放入其“被分配的机器”的队列
        for m in range(self.machine_count):
            machine_available[m] = 0
        total_ops = self.operationNum
        scheduled = 0

        for j in range(self.jobNum):
            if len(self.jobProcess[j]["operations"]) == 0:
                continue
            g0 = self._get_global_op_index(j, 0)
            m0 = machine_assignments[g0]["machine_id"]
            push_ready(m0, g0)

        # 主循环：直到所有工序都被派完
        while scheduled < total_ops:
            # 找“最早可用且队列非空”的机器
            candidate = None  # (avail_time, machine_id)
            for m, q in ready_by_machine.items():
                if q:  # 队列非空
                    t = machine_available.get(m, 0)
                    if (
                        (candidate is None)
                        or (t < candidate[0])
                        or (t == candidate[0] and m < candidate[1])
                    ):
                        candidate = (t, m)

            # 如果所有机器的就绪队列都空，但还有未完成工序：说明当前都在等前序完工
            # 这在设备驱动写法里通常“不会发生”，因为一旦完工我们会立刻把后继入队。
            # 为保险起见，兜底推进到“最早完工的机器时间”（但这里我们没有显式事件表，通常不需要）。
            if candidate is None:
                # 安全兜底：直接跳出（防死循环）。若发生，说明初始化没有把首工序入队。
                break

            _, m_sel = candidate
            # 取该机器最高优先级就绪工序
            neg_pr, gidx = heapq.heappop(ready_by_machine[m_sel])
            j, k_local = glb2jl(gidx)
            p_time = machine_assignments[gidx]["process_time"]

            st = max(machine_available[m_sel], job_last_end[j])
            et = st + p_time

            # 记录
            op_status[gidx] = (st, et, m_sel)
            machine_available[m_sel] = et
            job_last_end[j] = et
            scheduled += 1

            # 推入该job的下一道工序（若有）
            if k_local + 1 < len(self.jobProcess[j]["operations"]):
                g_next = self._get_global_op_index(j, k_local + 1)
                m_next = machine_assignments[g_next]["machine_id"]
                # 只有在“前序已完”的时刻才入队。此处前序刚完，立即入队。
                push_ready(m_next, g_next)

        # 目标值
        makespan = max((end for (_, end, _) in op_status.values()), default=0)

        # 统计负荷（稀疏版）
        load = {}
        for st, et, mid in op_status.values():
            load[mid] = load.get(mid, 0) + (et - st)
        max_load = max(load.values(), default=0)

        self.last_schedule = op_status
        return makespan, max_load

    def decode_chromosome(self, chrom):
        """
        设备驱动派工解码（非抢占）：
        返回 {machine_id: [(job_id, start, end, local_op_id), ...]}，每台机器内按 start 升序。
        """
        import heapq

        op_priority = chrom[: self.operationNum]
        machine_select = chrom[self.operationNum :]

        # 机器分配（按你的编码方式把每个全局工序分配到某台机器）
        machine_assignments = self._assign_machines(machine_select)

        # 设备可用时间、作业已完工时间
        machine_available = {m: 0 for m in range(self.machine_count)}
        job_last_end = {j: 0 for j in range(self.jobNum)}

        # 每台设备的就绪堆：m -> [(-priority, tie, gidx)]
        ready_by_machine = {}

        def push_ready(m, gidx):
            pr = float(op_priority[gidx])
            if m not in ready_by_machine:
                ready_by_machine[m] = []
            # 用负号变成最大堆效果；tie 用 gidx 保证稳定
            heapq.heappush(ready_by_machine[m], (-pr, gidx))

        # 辅助：从全局索引拿 job_id / 本地 op_id
        def glb2jl(gidx):
            info = self.op_mapping[gidx]
            return info["job_id"], info["op_id"]

        # 初始化：每个作业的第0道工序入它被分配到的设备队列
        for j in range(self.jobNum):
            if len(self.jobProcess[j]["operations"]) == 0:
                continue
            g0 = self._get_global_op_index(j, 0)
            m0 = machine_assignments[g0]["machine_id"]
            push_ready(m0, g0)

        # 派工循环
        total_ops = self.operationNum
        scheduled = 0
        # 记录：{gidx: (st, et, m, j, k_local)}
        op_status = {}

        while scheduled < total_ops:
            # 选“最早可用且就绪堆非空”的机器
            candidate = None  # (avail_time, machine_id)
            for m, q in ready_by_machine.items():
                if q:  # 该机有就绪工序
                    t = machine_available.get(m, 0)
                    if (
                        (candidate is None)
                        or (t < candidate[0])
                        or (t == candidate[0] and m < candidate[1])
                    ):
                        candidate = (t, m)
            if candidate is None:
                # 护栏：如果没有任何就绪工序，说明都在等前序完成（理论上不会出现）
                break

            _, m_sel = candidate
            # 取该机最高优先级就绪工序
            neg_pr, gidx = heapq.heappop(ready_by_machine[m_sel])
            j, k_local = glb2jl(gidx)
            p_time = machine_assignments[gidx]["process_time"]

            st = max(machine_available[m_sel], job_last_end[j])
            et = st + p_time

            op_status[gidx] = (st, et, m_sel, j, k_local)
            machine_available[m_sel] = et
            job_last_end[j] = et
            scheduled += 1

            # 把该 job 的下一道工序入队（若存在）
            if k_local + 1 < len(self.jobProcess[j]["operations"]):
                g_next = self._get_global_op_index(j, k_local + 1)
                m_next = machine_assignments[g_next]["machine_id"]
                push_ready(m_next, g_next)

        # 组装绘图结构：真实机器ID做 key
        schedule = {}
        for gidx, (st, et, m, j, k_local) in op_status.items():
            schedule.setdefault(m, []).append((j, st, et, k_local))

        # 每台机器按 start 排序
        for m in schedule:
            schedule[m].sort(key=lambda x: x[1])

        # 也把这次明细交给外部统计（比如检查机器14）
        self.last_schedule = {
            g: (st, et, m) for g, (st, et, m, _, _) in op_status.items()
        }
        return schedule

    def save_gantt(self, chrom, save_path):
        """
        根据染色体生成甘特图并保存
        y 轴直接用机器真实编号 (machine_id)，而不是行号。
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import os

        schedule = self.decode_chromosome(chrom)

        fig, ax = plt.subplots(figsize=(12, 6))

        # 给每个工件设置颜色
        jobs = sorted({job for ops in schedule.values() for (job, _, _, _) in ops})
        colors = plt.cm.tab20(np.linspace(0, 1, len(jobs)))
        job_color_map = {job: colors[i] for i, job in enumerate(jobs)}

        yticks, yticklabels = [], []
        for machine_id, tasks in schedule.items():
            yticks.append(machine_id)  # 直接用真实机器ID作为y
            yticklabels.append(f"M{machine_id}")  # 标签也是M{machine_id}
            for job, start, end, op in tasks:
                ax.barh(
                    machine_id,  # y = machine_id
                    end - start,
                    left=start,
                    color=job_color_map[job],
                    edgecolor="black",
                    height=0.8,
                )
                ax.text(
                    (start + end) / 2,
                    machine_id,
                    f"J{job}-O{op}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_title("FJSP Gantt Chart (machine ID as y-axis)")

        handles = [
            mpatches.Patch(color=job_color_map[j], label=f"Job {j}") for j in jobs
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Gantt chart saved: {save_path}")

    def _get_global_op_index(self, job_id, op_id):
        """获取全局操作索引"""
        # 计算该作业之前的操作总数
        prev_ops = 0
        for i in range(job_id):
            prev_ops += len(self.jobProcess[i]["operations"])
        return prev_ops + op_id

    def cal_ref_points(self, fitness_list):
        """计算参考点"""
        # 获取目标维度数
        ref_point = np.zeros(len(fitness_list[0]))

        # 遍历每个目标维度，计算最大值和均值
        for i in range(len(ref_point)):
            values = [fitness[i] for fitness in fitness_list]
            max_val = np.max(values)
            mean_val = np.mean(values)

            # 使用最大值加上一个动态调整的偏移量，确保参考点位于前沿解集外
            # 偏移量设置为最大值的10%或最大值与均值的差值的50%，以适应不同的情况
            offset = 0.1 * max_val  # 你可以根据需要调整偏移量的比例
            if max_val != mean_val:
                offset = max(
                    offset, 0.5 * abs(max_val - mean_val)
                )  # 更具鲁棒性，避免参考点太小

            ref_point[i] = max_val + offset

        # 将计算结果转为二维数组格式
        ref_points = np.array([ref_point])

        return ref_points
