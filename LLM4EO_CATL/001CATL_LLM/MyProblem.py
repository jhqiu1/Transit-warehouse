# -*- coding: utf-8 -*-
"""MyProblem.py"""
import copy
import random
import numpy as np
import geatpy as ea
import time
import json
import logging
import pickle

from pathlib import Path

import sys
import os

from aad_catl_search_operators.utils.problem_api import get_request, pyobj2json


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, obj_list, batchNo, logdir, obj_weight_vector):
        name = "MyProblem"  # 初始化name(函数名称,可以随意设置)
        self.obj_list = obj_list
        self.obj_weight_vector = obj_weight_vector
        self.ref_points = []
        self.M = len(obj_list)  # 初始化M(目标维数)
        maxormins = [1] * self.M  # 初始化目标最小最大化标记列表,1:min；-1:max

        self.logdir = logdir  # 日志记录

        self.batchNo = batchNo

        data = self._ini_data()
        self.data = data

        Dims = data["Dims"]  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dims  # 初始化决策变量类型， 0： 连续； 1： 离散
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

    def _ini_data(self):
        # 确定存储目录和路径
        base_dir = Path(__file__).parent  # 获取当前代码文件所在目录
        instances_dir = base_dir / "Instances"
        instances_dir.mkdir(exist_ok=True)  # 创建Instances目录（如果不存在）
        instance_name = str(self.batchNo) + ".pkl"
        file_path = instances_dir / instance_name
        # 保存
        if file_path.exists():
            with open(file_path, "rb") as f:  # 读取
                instance = pickle.load(f)
        else:
            instance = self.get_init_data()
            with open(file_path, "wb") as f:  # 写入
                pickle.dump(instance, f)
        return instance

    def get_init_data(self):
        url1 = "http://10.59.5.176:9091/init_batch_no"  # 修正URL
        request_dict1 = {"batch_no": self.batchNo}  # 修正参数名
        request_json1 = pyobj2json(request_dict1)
        # 调用接口
        start_time = time.time()
        code, data, msg = get_request(url1, request_json1)
        elapsed_time = time.time() - start_time

        # todo: msg错误处理
        if msg != "成功":
            # 调用封装好的日志方法
            self._log_api_call(
                url1, request_dict1, request_json1, code, data, msg, elapsed_time
            )
            raise ValueError("get_init_data failed")
        return data

    def aimFunc(self, pop):  # 目标函数、pop为传入的种群对象
        # 得到决策变量矩阵
        Vars = pop.Phen
        # pop.cV=[]
        pop.ObjV = np.zeros((len(Vars), self.M))
        for i in range(len(Vars)):
            # pop.ObjV[i] = decode(Vars[i])[-1]
            # 变量返回
            obj_dict = self.calcFunc(Vars[i])
            for j, obj_name in enumerate(self.obj_list):
                pop.ObjV[i][j] = obj_dict[obj_name]

    def calcFunc(self, x):
        url2 = "http://10.59.5.176:9091/main_cal_phen"
        request_dict2 = {
            "batch_no": self.batchNo,  # 修正参数名
            "solution": x.astype(int).tolist(),
        }
        request_json = pyobj2json(request_dict2)
        # print(request_json)
        start_time = time.time()
        code, data, msg = get_request(url2, request_json)
        elapsed_time = time.time() - start_time

        if msg != "成功":
            # 调用封装好的日志方法
            self._log_api_call(
                url2, request_dict2, request_json, code, data, msg, elapsed_time
            )
            raise ValueError("calcFunc failed")
        # ini ref point
        if len(self.ref_points) > 0:
            weighted_obj = self.weighted_obj(data)
        else:
            weighted_obj = data
        return weighted_obj

    def weighted_obj(self, obj_dict):
        weighted_sum = 0
        for i in range(len(self.obj_list)):
            weighted_sum += self.obj_weight_vector[i] * (
                (obj_dict[self.obj_list[i]] - self.ref_porefint[i]) / self.ref_point[i]
            )

        return weighted_sum

    def _log_api_call(
        self, url, request_dict, request_json, code, data, msg, elapsed_time
    ):
        """
        记录API调用日志（JSON Lines格式）
        参数:
            url: API地址
            request_dict: 请求参数字典
            request_json: JSON序列化后的请求体
            code: 响应状态码
            data: 响应数据
            msg: 响应消息
            elapsed_time: 请求耗时(秒)
        """
        # 1. 构建日志条目
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request": {
                "url": url,
                "parameters": request_dict,
                "json_payload": request_json,
            },
            "response": {
                "status_code": code,
                "data": data,
                "message": msg,
                "elapsed_seconds": round(elapsed_time, 4),
            },
        }
        # 2. 确保日志目录存在
        os.makedirs(self.logdir, exist_ok=True)
        log_file = os.path.join(self.logdir, "api_log.json")

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.error(f"API日志写入失败: {e}")

    # def calcFunc(self, x):
    #
    #     # pop.Objv[i]=decode(Vars[i])[-1]
    #     # 变量返回
    #     start_time, end_time, start_time_wait, end_time_wait, job_ope_st, job_ope_et, job_ope_plan, constraint, \
    #         JM, M_j_st, M_j_et, record_machine = decode(x, self.data)
    #
    #     # note: coding evaluator
    #     # kwargs = {
    #     #     'start_time': start_time,
    #     #     'end_time': end_time,
    #     #     'start_time_wait': start_time_wait,
    #     #     'end_time_wait': end_time_wait,
    #     #     'job_ope_st': job_ope_st,
    #     #     'job_ope_et': job_ope_et,
    #     #     'job_ope_plan': job_ope_plan,
    #     #     'constraint': constraint,
    #     #     'JM': JM,
    #     #     'M_j_st': M_j_st,
    #     #     'M_j_et': M_j_et,
    #     #     'record_machine': record_machine
    #     # }
    #     # evaluator = Evaluator(self.data, **kwargs)
    #     # evaluator.evaluate_constraints()
    #
    #     # #计算目标
    #     # 目标1.计算延期的数量EPS
    #     # 遍历每个项目最后一道工序的日排产计划和数量
    #     obj_delay_job_num = 0
    #     obj_delay_job_day = 0
    #     for job, operation in self.data.job_operation.items():
    #         job_plan_temp = job_ope_plan[job][operation[-1]]
    #         job_ope_st_temp = job_ope_st[job][operation[-1]]
    #         # todo ==========================
    #         job_ope_et_temp = job_ope_et[job][operation[-1]]
    #         job_plan_day_temp = {job_ope_et_temp + index: job_plan_temp[index] for index in
    #                              range(len(job_plan_temp))}
    #         delay_job_num = 0
    #         for key, value in job_plan_day_temp.items():
    #             if key > job_deadline[job]:
    #                 delay_job_num += value * job_priority[job]
    #         obj_delay_job_num += delay_job_num
    #
    #         # 目标2,计算每个项目的延期时间
    #         job_max_end = max(get_last_value(job_ope_et[job]))
    #         delay_job_day = max(job_max_end - job_deadline[job], 0) * job_priority[
    #             job]
    #         obj_delay_job_day += delay_job_day
    #         # todo ===========================
    #
    #     # 目标1 2针对的是EPR订单最早开始时间
    #     obj_min_start_time = 0
    #     for job, operation in self.data.job_operation.items():
    #         job_min_start = min(get_last_value(job_ope_st[job]))
    #         obj_min_start_time += job_min_start * job_priority[job]
    #     # 任务分配尽量集中资源(前工序),跨区域／拉线次数少
    #     switch_area_num = 0
    #     for job, m_ in JM.items():
    #         m_list_ = [m__[0] for m__ in m_.values()]
    #         area_list_ = [machine_area[m_] for m_ in m_list_]
    #         switch_area_num += count_switch_regions(area_list_)
    #
    #     # for job,m_in JM.items():
    #     # 计算切拉时间
    #     switch_num = 0
    #     for m, v in record_machine.items():
    #         # if m not in epr_machine_index:
    #         #     continue
    #         # 切拉列表
    #         switch_time_job_list = record_machine[m]  # [keys[idx] for idx in machine_job_idx]
    #         for index_job in range(len(switch_time_job_list) - 1):
    #             # 1+1项目切i项目
    #             if switch_job_dict[switch_time_job_list[index_job]] != switch_job_dict[switch_time_job_list[index_job + 1]]:
    #                 switch_num += 1
    #         # print（机器，m，项目i'，i，项目i+1，i+1，切拉时间'，switch_time_matrix[ml[i+1][i]）
    #
    #     # 计算开机数量
    #     used_m_num = 0
    #     for mi in start_time:
    #         if np.where(start_time[mi] > 0)[0].shape[0] > 0:
    #             used_m_num += 1
    #
    #     return {
    #         "obj_delay_job_day": obj_delay_job_day,
    #         "obj_delay_job_num": obj_delay_job_num,
    #         "obj_min_start_time": obj_min_start_time,
    #         "used_m_num": used_m_num,
    #         "switch_area_num": switch_area_num,
    #         "switch_num": switch_num,
    #     }

    # def calcFunc(self, x):  # 目标函数、pop为传入的种群对象
    #
    #     start_time, end_time, start_time_wait, end_time_wait, job_ope_st, job_ope_et, job_ope_plan, constraint, \
    #         JM, M_j_st, M_j_et, record_machine = decode(x, self.data)
    #
    #     # note: coding evaluator
    #     # kwargs = {
    #     #     'start_time': start_time,
    #     #     'end_time': end_time,
    #     #     'start_time_wait': start_time_wait,
    #     #     'end_time_wait': end_time_wait,
    #     #     'job_ope_st': job_ope_st,
    #     #     'job_ope_et': job_ope_et,
    #     #     'job_ope_plan': job_ope_plan,
    #     #     'constraint': constraint,
    #     #     'JM': JM,
    #     #     'M_j_st': M_j_st,
    #     #     'M_j_et': M_j_et,
    #     #     'record_machine': record_machine
    #     # }
    #     # evaluator = Evaluator(self.data, **kwargs)
    #     # evaluator.evaluate_constraints()
    #
    #     # #计算目标
    #     # 目标1.计算延期的数量EPS
    #     # 遍历每个项目最后一道工序的日排产计划和数量
    #     obj_delay_job_num = 0
    #     obj_delay_job_day = 0
    #     for job, operation in self.data.job_operation.items():
    #         job_plan_temp = job_ope_plan[job][operation[-1]]
    #         job_ope_st_temp = job_ope_st[job][operation[-1]]
    #         # todo ==========================
    #         job_ope_et_temp = job_ope_et[job][operation[-1]]
    #         job_plan_day_temp = {job_ope_et_temp + index: job_plan_temp[index] for index in
    #                              range(len(job_plan_temp))}
    #         delay_job_num = 0
    #         for key, value in job_plan_day_temp.items():
    #             if key > self.data.packDueDateDict[job]:
    #                 delay_job_num += value * self.data.packPriorityDict[job]
    #         obj_delay_job_num += delay_job_num
    #
    #         # 目标2,计算每个项目的延期时间
    #         job_max_end = max(get_last_value(job_ope_et[job]))
    #         delay_job_day = max(job_max_end - self.data.packDueDateDict[job], 0) * self.data.packPriorityDict[
    #             job]
    #         obj_delay_job_day += delay_job_day
    #         # todo ===========================
    #
    #     # 目标1 2针对的是EPR订单最早开始时间
    #     obj_min_start_time = 0
    #     for job, operation in self.data.job_operation.items():
    #         job_min_start = min(get_last_value(job_ope_st[job]))
    #         obj_min_start_time += job_min_start * self.data.packPriorityDict[job]
    #     # 任务分配尽量集中资源(前工序),跨区域／拉线次数少
    #     switch_area_num = 0
    #     for job, m_ in JM.items():
    #         m_list_ = [m__[0] for m__ in m_.values()]
    #         area_list_ = [self.data.machine_area[m_] for m_ in m_list_]
    #         switch_area_num += count_switch_regions(area_list_)
    #
    #     # for job,m_in JM.items():
    #     # 计算切拉时间
    #     switch_num = 0
    #     for m, v in record_machine.items():
    #         # if m not in epr_machine_index:
    #         #     continue
    #         # 切拉列表
    #         switch_time_job_list = record_machine[m]  # [keys[idx] for idx in machine_job_idx]
    #         for index_job in range(len(switch_time_job_list) - 1):
    #             # 1+1项目切i项目
    #             if self.data.switch_job_dict[switch_time_job_list[index_job]] != self.data.switch_job_dict[switch_time_job_list[index_job + 1]]:
    #                 switch_num += 1
    #         # print（机器，m，项目i'，i，项目i+1，i+1，切拉时间'，switch_time_matrix[ml[i+1][i]）
    #
    #     # 计算开机数量
    #     used_m_num = 0
    #     for mi in start_time:
    #         if np.where(start_time[mi] > 0)[0].shape[0] > 0:
    #             used_m_num += 1
    #
    #     # EPR
    #     # if self.obj_flag == 3:  # EPS
    #     #     pop.objv[i][0] = obj_delay_job_num
    #     #     pop.objv[i][1] = switch_num
    #     # elif self.obj_flag == 2:  # EPR
    #     #     pop.objv[i][0] = obj_min_start_time
    #     #     pop.objv[i][1] = switch_num
    #     # elif self.obj_flag == 1:  # EPR和EPS
    #     # pop.ObjV[i][0] = obj_delay_job_num
    #     return {
    #         "obj_delay_job_day": obj_delay_job_day,
    #         "obj_delay_job_num": obj_delay_job_num,
    #         "obj_min_start_time": obj_min_start_time,
    #         "used_m_num": used_m_num,
    #         "switch_area_num": switch_area_num,
    #         "switch_num": switch_num,
    #         # "total_priority_time": total_priority_time
    #     }
    # pop.objv[i][0] = obj_min_start_time
    # pop.objv[i][1] = obj_delay_job_num
    # pop.objv[i][2] = switch_num

    # 物料连续生产约束
    # 工序连续生产约束
    # 工艺时间限制约束
    # 目标函数最少延期数
    # 目标函数最少开机数
    # pop.cV.append（constraint)

    # pop.ObjV[i][0] = obj_min_start_time
    # pop.ObjV[i][1] = switch_num
