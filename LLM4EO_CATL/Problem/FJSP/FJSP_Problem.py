import numpy as np
import random
import copy
import geatpy as ea  # 导入geatpy库
import os
import json


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
        # varTypes = [0] * Dims  # 初始化决策变量类型，0：连续；1：离散
        varTypes = data[
            "varTypes"
        ]  # 初始化决策变量类型，0：连续；1：离散 (连续+整数编码)
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
        print("Creating initial data...", instance_path)
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
        # 初始化种群矩阵
        population = np.zeros((pop, self.Dims))
        fit_results = []

        for i in range(pop):
            op_start = 0
            # 工序优先级部分：随机生成，但保持同一作业内操作顺序
            for job in self.jobProcess:
                num_ops = len(job["operations"])
                # 生成随机优先级
                priorities = np.random.rand(num_ops)
                # 按升序排序，确保操作顺序正确（先执行的优先级小）
                priorities.sort()
                population[i, op_start : op_start + num_ops] = priorities
                op_start += num_ops

            # 机器选择部分：随机分配到有效机器
            for j in range(self.operationNum):
                op_info = self.op_mapping[j]
                num_machines = len(op_info["machines"])
                population[i, self.operationNum + j] = np.random.randint(
                    0, num_machines
                )

            # 计算适应度
            fit_results.append(self.calcFunc(population[i, :]))

        # 储存初始种群和适应度
        self.ini_population = population
        self.ini_fit_results = fit_results
        self.ref_points = self.read_ref_points(fit_results).tolist()

        return fit_results

    def calcFunc(self, x):
        op_priority = x[: self.operationNum]
        machine_select = x[self.operationNum :]
        op_sequences = self._get_op_sequences(op_priority)
        machine_assignments = self._assign_machines(machine_select)
        makespan, max_load, total_load = self._build_schedule(
            op_sequences, machine_assignments, op_priority
        )
        return makespan, total_load  # 返回最大负荷

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
            makespan, total_load = self.calcFunc(Vars[i])
            pop.ObjV[i, 0] = makespan
            pop.ObjV[i, 1] = total_load

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

            if candidate is None:
                break  # 安全兜底：避免死循环

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
                push_ready(m_next, g_next)

        # === 目标计算 ===
        # 1. makespan
        makespan = max((end for (_, end, _) in op_status.values()), default=0)

        # 2. 最大设备负荷 & 3. 总设备负荷
        load = {}
        for st, et, mid in op_status.values():
            load[mid] = load.get(mid, 0) + (et - st)
        max_load = max(load.values(), default=0)
        total_load = sum(load.values())

        self.last_schedule = op_status
        return makespan, max_load, total_load

    def _get_global_op_index(self, job_id, op_id):
        """获取全局操作索引"""
        # 计算该作业之前的操作总数
        prev_ops = 0
        for i in range(job_id):
            prev_ops += len(self.jobProcess[i]["operations"])
        return prev_ops + op_id

    def read_ref_points(self, fitness_list):
        """优先读取 self.batchNo 对应的 refpoint.json，没有则计算并保存"""
        base, _ = os.path.splitext(self.batchNo)
        json_path = base + "_refpoint.json"
        obj_list_sorted = sorted(self.obj_list)
        obj_list_str = "_".join(obj_list_sorted)

        # 如果文件已存在，直接读取
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                # 读取整个JSON文件内容，它是一个字典
                all_data = json.load(f)
                # 检查当前目标组合的参考点是否存在
                if obj_list_str in all_data:
                    # 如果存在，返回对应的参考点
                    return np.array(all_data[obj_list_str])
                else:
                    # 如果不存在，则计算并保存，同时保留其他目标组合的数据
                    return self.cal_ref_points(
                        json_path, fitness_list, obj_list_str, all_data
                    )
        else:
            # 如果文件不存在，直接计算，初始数据为空字典
            return self.cal_ref_points(json_path, fitness_list, obj_list_str, {})

    def cal_ref_points(self, json_path, fitness_list, obj_list_str, existing_data=None):
        """
        为特定的目标列表（obj_list_str）计算参考点并保存到文件
        existing_data: 文件中已存在的其他目标组合的参考点数据（字典）
        """
        if existing_data is None:
            existing_data = {}

        # 计算参考点的逻辑保持不变
        ref_point = []
        for i in range(len(fitness_list[0])):
            values = [fitness[i] for fitness in fitness_list]
            max_val = np.max(values)
            mean_val = np.mean(values)
            offset = (
                max(0.1 * max_val, 0.5 * abs(max_val - mean_val))
                if max_val != mean_val
                else 0.1 * max_val
            )
            ref_point.append(max_val + offset)

        ref_points = np.array([ref_point])

        # 更新数据字典：添加或更新当前目标组合的参考点
        existing_data[obj_list_str] = ref_points.tolist()

        # 保存到 json：将整个更新后的字典写回文件
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

        return ref_points
