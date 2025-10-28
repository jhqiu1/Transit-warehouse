import numpy as np
import json
import os
import geatpy as ea
from llm4ad.base.code import TextFunctionProgramConverter
from typing import List, Any, Callable
from typing import Tuple

# 使用globals()获取全局命名空间，但添加必要的模块
all_globals_namespace = globals().copy()
all_globals_namespace.update({"np": np, "ea": ea, "List": List})


import shutil


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


def load_best_operators_from_json(json_path):
    """
    从JSON文件加载最佳算子代码

    参数:
        json_path: JSON文件路径

    返回:
        包含最佳算子代码的字典
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # 找到分数最高的记录
    best_record = max(data, key=lambda x: x["score"])
    return best_record["operators"]


def function_string_to_program(code_string: str, function_name: str) -> Callable:
    """
    将代码字符串转换为可执行函数

    参数:
        code_string: 包含函数定义的代码字符串
        function_name: 要提取的函数名称

    返回:
        可执行函数
    """
    # 创建局部命名空间
    local_namespace = {}

    # 执行代码字符串
    exec(code_string, globals(), local_namespace)

    # 返回指定的函数

    return local_namespace[function_name]


def function_json_to_program(path: str) -> Callable:
    """
    从JSON文件中读取函数代码并转换为可执行函数

    Args:
        path: 包含samples_best.json文件的目录路径

    Returns:
        可执行的Python函数
    """
    path1 = os.path.join(path, "samples", "samples_best.json")
    print(f"Loading function from: {path1}")

    try:
        with open(path1, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            entry = json_data[-1]  # 获取最后一个条目
            function_code = entry.get("function")
            return function_to_callable(function_code)

    except FileNotFoundError:
        print(f"Error: File not found at {path1}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {path1}")
        raise
    except Exception as e:
        print(f"Error processing function: {e}")
        raise


def function_to_callable(function_code):
    # 转换为程序对象
    program = TextFunctionProgramConverter.text_to_program(function_code)
    program_str = str(program)
    print("Program code:")
    print(program_str)

    # 提取函数名
    function_name = TextFunctionProgramConverter.text_to_function(program_str).name

    # 执行程序代码
    exec(program_str, all_globals_namespace)

    # 获取函数引用
    program_callable = all_globals_namespace.get(function_name)

    if program_callable is None:
        raise ValueError(f"Function '{function_name}' not found in executed code")

    return program_callable


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


if __name__ == "__main__":
    try:
        func = function_json_to_program(
            "../PLS_LLM/generalization/data1_顺序/PLS@eoh@assignment"
        )
        print(f"Successfully loaded function: {func.__name__}")
    except Exception as e:
        print(f"Failed to load function: {e}")
