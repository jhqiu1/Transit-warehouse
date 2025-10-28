import argparse
from typing import Any
import shutil

import wandb, numpy as np
import requests
import json
import re
from llm4ad.base import (
    Evaluation,
    SecureEvaluator,
    LLM,
    TextFunctionProgramConverter as TFPC,
)
from llm4ad.method.hillclimb import HillClimb
from llm4ad.method.eoh import EoH
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.tools.profiler import WandBProfiler, TensorboardProfiler
from openai import OpenAI

from NSGA3_LLM import nsga_llm

from _evaluate import (
    MyEvaluation,
    generate_ref_point,
    perm_mutation_template,
    assign_mutation_template,
    assign_crossover_template,
    perm_crossover_template,
)
from aad_catl_search_operators.CATL_LLM.MyProblem import MyProblem
from aad_catl_search_operators.utils.path_util import get_path

import os


class DeepSeekV3API(LLM):

    def __init__(self, trim=False):
        super().__init__()
        self._trim = trim

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        while True:
            try:
                response = self._do_request(prompt)
                if self._trim:
                    response = self._trim_response(response)
                return response
            except Exception as e:
                print(e)
                continue

    def _do_request(self, content: str) -> str:
        if isinstance(content, str):
            content = [{"role": "user", "content": content.strip()}]
        else:
            content = content.strip()
        # content = [{'role': 'system', 'content': 'Assume you are a code completion model.'}, {'role': 'user', 'content': content}]
        client = OpenAI(api_key="xxxx", base_url="http://10.38.60.151:7891/v1")

        response = client.chat.completions.create(
            model="qwen2.5-32b",
            messages=content,
            stream=False,
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content

    def _trim_response(self, response: str):
        from llm4ad.base import TextFunctionProgramConverter as TFPC
        import re

        match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if match:
            extracted_text = match.group(1)
            prog = TFPC.text_to_program(extracted_text)
            last_func = prog.functions[-1]
            return str(last_func)
        else:
            extracted_text = response
        return extracted_text


# parser = argparse.ArgumentParser()
# parser.add_argument('--run', type=int)
# args = parser.parse_args()


def manage_directory(path):
    """
    检查指定的文件夹是否存在，如果存在则删除，然后重新创建。
    如果文件夹不存在，则直接创建。

    :param path: 文件夹路径
    """
    if os.path.exists(path):
        # 如果文件夹存在，先删除
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")

    # 重新创建文件夹
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")


if __name__ == "__main__":

    ###########################################
    data_list = ["PS_20250819191817"]

    template_dict = {
        "permutation_crossover": perm_crossover_template,
        "permutation_mutation": perm_mutation_template,
        "assignment_crossover": assign_crossover_template,
        "assignment_mutation": assign_mutation_template,
    }

    opt_obj_list = ["obj_1", "obj_2", "obj_3", "obj_4", "obj_5"]  # 最终优化的目标
    # api_obj_list = ['obj_1', 'obj_2', 'obj_3', 'obj_4', 'obj_5']  # 接口返回的目标
    obj_weight_vector = [10, 0.9, 0.8, 0.7, 0.6]
    ###########################################
    # num_evaluators = 6
    # num_samplers = 6
    # max_sample_nums = 54 #沿用
    # generation_num = 20 #GA迭代次数
    # debug para
    num_evaluators = 6  # 验证并行算子评估器的进程数量
    num_samplers = 6  # 并行调LLM采样器的数量
    max_sample_nums = (
        54  # max_sample_nums 大模型生成算子的个数 达到这个个数或者是迭代次数都会停止
    )
    generation_num = 20  # GA迭代次数 每次评估的运行代数
    exp_dir = "six_obj"
    log_dir = "aad_catl_search_operators/CATL_LLM/" + exp_dir + "/{}"

    ###########################################

    for data in data_list:

        problem = MyProblem(opt_obj_list, data, exp_dir, obj_weight_vector)
        problem.ref_points = generate_ref_point(problem, 10)

        for operator, template in template_dict.items():
            log_path = str(get_path(log_dir.format(operator)))
            manage_directory(log_path)

            profiler = TensorboardProfiler(
                wandb_project_name="aad-catl",
                log_dir=log_path,
                name=f"CATL@eoh@nsga_{operator}",
                create_random_path=False,
            )
            ## HillClimb 必须初始template合法
            eoh = HillClimb(
                evaluation=MyEvaluation(
                    template=template,
                    operator_name=operator,
                    algorithm=nsga_llm,
                    problem=problem,
                    ref_point=problem.ref_points,
                    generation_num=generation_num,
                ),
                num_evaluators=num_evaluators,  # 指定线程池的最大工作线程数，即同时执行评估任务的线程数量上限
                num_samplers=num_samplers,
                max_sample_nums=max_sample_nums,
                profiler=profiler,
                # llm=GPT4oMini(),
                # llm=DGX2API(),
                llm=DeepSeekV3API(),
                debug_mode=True,
            )
            eoh.run()
