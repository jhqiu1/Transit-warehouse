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
        varTypes = [0] * Dims  # 初始化决策变量类型，0：连续；1：离散
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
        self.gene_ini_pops()  # 初始化种群

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
        lbs = [
            0 for _ in range(self.jobNum + self.operationNum * 2)
        ]  # 设备顺序与job一致
        ubs = [
            1 for _ in range(self.jobNum + self.operationNum * 2)
        ]  # 设备顺序与job一致
        MS_len = self.operationNum
        eadata = {}
        eadata["lbs"] = lbs
        eadata["ubs"] = ubs
        eadata["Dims"] = self.jobNum + self.operationNum * 2
        eadata["MS_len"] = MS_len
        return eadata

    def gene_ini_pops(self, pop=10):
        # 随机生成初始总群，在决策变量范围内随机构建种群
        population = np.zeros((pop, self.Dims))
        fit_results = []
        for i in range(pop):
            for j in range(self.Dims):
                if self.varTypes[j] == 1:  # 离散变量
                    population[i, j] = np.random.randint(
                        low=self.lbs[j], high=self.ubs[j] + 1  # 上界包含
                    )
                else:  # 连续变量
                    population[i, j] = np.random.uniform(
                        low=self.lbs[j], high=self.ubs[j]
                    )
            fit_results.append(self.calcFunc(population[i, :]))
        # 储存初始种群和适应度
        self.ini_population = population
        self.ini_fit_results = fit_results
        return fit_results

    def calcFunc(self, x):
        """
        计算单个pop的makespan和设备利用率
        :param x: 染色体向量
        :return: (makespan, utilization)
        """
        # 1. 解析染色体各部分
        job_priority = x[: self.jobNum]  # 作业优先级
        op_priority = x[self.jobNum : self.jobNum + self.operationNum]  # 操作优先级
        machine_select = x[self.jobNum + self.operationNum :]  # 机器选择值

        # 2. 确定作业加工顺序
        job_order = self._get_job_order(job_priority)

        # 3. 确定每个作业内部的操作顺序
        op_sequences = self._get_op_sequences(job_order, op_priority)

        # 4. 为每个操作分配机器
        machine_assignments = self._assign_machines(machine_select)

        # 5. 构建调度方案并计算目标值
        makespan, utilization = self._build_schedule(
            job_order, op_sequences, machine_assignments
        )

        return makespan, utilization

    def _get_job_order(self, job_priority):
        """根据作业优先级确定作业加工顺序"""
        # 按优先级降序排序（优先级越高越先加工）
        sorted_indices = np.argsort(job_priority)[::-1]
        return sorted_indices.tolist()

    def _get_op_sequences(self, job_order, op_priority):
        """确定每个作业内部的操作顺序"""
        op_sequences = {}

        # 为每个作业构建操作列表
        op_start = 0
        for job_id in range(self.jobNum):
            job = self.jobProcess[job_id]
            num_ops = len(job["operations"])

            # 获取该作业的操作优先级部分
            job_op_priority = op_priority[op_start : op_start + num_ops]
            op_start += num_ops

            # 按优先级降序排序（优先级越高越先加工）
            sorted_indices = np.argsort(job_op_priority)[::-1]
            op_sequences[job_id] = sorted_indices.tolist()

        return op_sequences

    def _assign_machines(self, machine_select):
        """为每个操作分配机器"""
        assignments = []

        for i in range(self.operationNum):
            op_info = self.op_mapping[i]
            available_machines = list(op_info["machines"].keys())

            # 对机器进行排序以确保一致性
            available_machines.sort()

            # 根据机器选择值确定使用哪台机器
            select_value = machine_select[i]
            num_machines = len(available_machines)

            # 将[0,1]区间划分为num_machines个等长子区间
            interval_size = 1.0 / num_machines
            machine_idx = min(int(select_value / interval_size), num_machines - 1)

            machine_id = available_machines[machine_idx]
            process_time = op_info["machines"][machine_id]

            assignments.append({"machine_id": machine_id, "process_time": process_time})

        return assignments

    def _build_schedule(self, job_order, op_sequences, machine_assignments):
        """构建调度方案并计算目标值（修正版）"""
        # 初始化数据结构
        machine_available_time = [0] * self.machine_count  # 每台机器的可用时间
        job_last_completion = [0] * self.jobNum  # 每个作业最后完成时间
        op_scheduled = [False] * self.operationNum  # 操作是否已调度
        op_start_times = [0] * self.operationNum
        op_end_times = [0] * self.operationNum

        # 总加工时间（用于计算设备利用率）
        total_processing_time = 0

        # 构建全局操作优先级列表
        global_ops = []
        op_idx = 0
        for job_id in range(self.jobNum):
            job = self.jobProcess[job_id]
            for op_id in range(len(job["operations"])):
                # 计算作业优先级（在job_order中的位置）
                job_priority = (
                    self.jobNum - job_order.index(job_id) if job_id in job_order else 0
                )

                # 计算操作优先级（在op_sequences中的位置）
                op_priority = (
                    len(op_sequences[job_id]) - op_sequences[job_id].index(op_id)
                    if op_id in op_sequences[job_id]
                    else 0
                )

                # 组合优先级（确保作业优先级权重更高）
                priority = job_priority * 1000 + op_priority

                global_ops.append(
                    {
                        "global_idx": op_idx,
                        "job_id": job_id,
                        "op_id": op_id,
                        "priority": priority,
                    }
                )
                op_idx += 1

        # 按优先级降序排序（优先级高的先调度）
        global_ops.sort(key=lambda x: x["priority"], reverse=True)

        # 调度所有操作
        scheduled_count = 0
        while scheduled_count < self.operationNum:
            # 选择优先级最高的可调度操作
            for op_info in global_ops:
                if op_scheduled[op_info["global_idx"]]:
                    continue

                job_id = op_info["job_id"]
                op_id = op_info["op_id"]
                global_idx = op_info["global_idx"]

                # 检查前序操作是否完成（作业约束）
                if op_id > 0:
                    prev_op_global_idx = self._get_global_op_index(job_id, op_id - 1)
                    if not op_scheduled[prev_op_global_idx]:
                        continue

                # 获取机器分配信息
                machine_assignment = machine_assignments[global_idx]
                machine_id = machine_assignment["machine_id"]
                process_time = machine_assignment["process_time"]

                # 计算最早开始时间
                # 作业约束：必须在前一个操作完成后
                if op_id == 0:
                    job_start = job_last_completion[job_id]
                else:
                    prev_op_global_idx = self._get_global_op_index(job_id, op_id - 1)
                    job_start = op_end_times[prev_op_global_idx]

                # 机器约束：必须在机器空闲时
                machine_start = machine_available_time[machine_id]

                start_time = max(job_start, machine_start)

                # 调度该操作
                end_time = start_time + process_time
                op_start_times[global_idx] = start_time
                op_end_times[global_idx] = end_time
                op_scheduled[global_idx] = True
                scheduled_count += 1

                # 更新进度
                job_last_completion[job_id] = end_time
                machine_available_time[machine_id] = end_time
                total_processing_time += process_time

                # 跳出内层循环，重新选择最高优先级操作
                break

        # 计算makespan（最大完成时间）
        makespan = max(job_last_completion)

        # 计算设备利用率
        total_machine_time = makespan * self.machine_count
        utilization = (
            total_processing_time / total_machine_time if total_machine_time > 0 else 0
        )

        return makespan, utilization

    def _get_global_op_index(self, job_id, op_id):
        """获取全局操作索引"""
        # 计算该作业之前的操作总数
        prev_ops = 0
        for i in range(job_id):
            prev_ops += len(self.jobProcess[i]["operations"])
        return prev_ops + op_id
