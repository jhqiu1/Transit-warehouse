import random
import math
import matplotlib.pyplot as plt
import os
import time
import inspect
from copy import deepcopy
import glob
import datetime

from LLM.Prompts import Prompts
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH
from llm4ad.tools.profiler import WandBProfiler, TensorboardProfiler
from _evaluate import (
    MyEvaluation,
    # ma_crossover_template,
    # ma_mutation_template,
    # op_crossover_template,
    # op_mutation_template,
    # op_crossover,
    # op_mutation,
    # ma_crossover,
    # ma_mutation,
)

from _history_operators import (
    ma_crossover_template,
    ma_mutation_template,
    op_crossover_template,
    op_mutation_template,
    op_crossover,
    op_mutation,
    ma_crossover,
    ma_mutation,
)
from utils.warmup_cache import (
    load_warmup_batch,
    _serialize_population_simple,
    save_warmup_batch,
)

# from _storage import GlobalResultStorage
from typing import List, Dict, Callable, Optional, Any
from Problem.FJSP.FJSP_Problem import FJSP_Problem
from Algrithm.NSGA2 import NSGA_FJSP
from logger.llm4eo_logger import ExperimentLogger
from utils.code_json_to_program import manage_directory
from utils.result_save import save_operators_results

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_path(*relative_parts):
    return os.path.join(PROJECT_ROOT, *relative_parts)


# CD eva
def quick_oc_eva(main_args, problem, dir, state_variables):
    # update para.
    max_cycles = main_args["max_cycles"]
    operators_to_optimize = main_args["operators_to_optimize"]
    template_dict = state_variables
    current_operators = deepcopy(main_args["default_operators_source"])

    best_score = 0
    for cycle in range(max_cycles):

        print(f"--- Starting coordinate cycle {cycle+1}/{max_cycles} ---")
        # 依次优化每个算子
        for state_index in range(len(operators_to_optimize)):
            operator_name = operators_to_optimize[state_index]
            print(f"Optimizing {operator_name}...")
            template = template_dict[state_index]

            # 创建针对当前算子的评估器
            operator_evaluator = MyEvaluation(
                algorithm=NSGA_FJSP,
                problem=problem,
                exp_name=exp_name,
                template=template,
                # template=default_operators_source[operator_name],
                ev_operator_name=operator_name,
                ref_point=problem.ref_points,
                generation_num=main_args["generation_num"],
                n_evals=main_args["n_evals"],
                pop_size=main_args["pop_size"],
            )
            # 更新当前operators set
            operator_evaluator.save_operatorstr_bydict(current_operators)
            # logger set
            log_path = str(
                get_path(log_dir.format(f"cycle{dir}_{cycle+1}_{operator_name}"))
            )
            manage_directory(log_path)
            profiler = TensorboardProfiler(
                wandb_project_name="aad-catl",
                log_dir=log_path,
                name=f"CATL@eoh@nsga_{operator_name}_cycle{cycle+1}",
                create_random_path=False,
            )
            # 运行EOH优化当前算子
            eoh = EoH(
                evaluation=operator_evaluator,
                profiler=profiler,
                llm=llm,
                debug_mode=True,
                max_generations=main_args["eoh_max_generations"],
                max_sample_nums=main_args["eoh_max_sample_nums"],
                pop_size=main_args["eoh_pop_size"],
                num_samplers=main_args["eoh_num_samplers"],
                num_evaluators=main_args["eoh_num_evaluators"],
            )
            # run eoh for searching this operator
            eoh.run()
            ope_pops = eoh._population
            # 获取最佳函数及其分数
            if ope_pops and len(ope_pops.population) > 0:  # 确保种群不为空
                best_function = ope_pops.population[0]  # 第一个个体就是分数最高的
                best_score_val = best_function.score
                # 检查分数是否有效
                if (
                    best_score_val is not None
                    and best_score_val > float("-inf")
                    and best_score_val > best_score
                ):
                    best_score = best_score_val
                    # 更新当前算子为源代码字符串
                    current_operators[operator_name] = str(best_function)

    return current_operators, best_score


class temaplte_storage:
    def __init__(self, origin_template, llm):
        self.origin_template = origin_template
        self.teme_storage = {}
        self.best_pop = {}
        self.ini_storage(origin_template)
        self.Prompt = Prompts()
        self.llm = llm
        self.pri_operators = []
        self.output_dir = main_args["exp_dir"]

    def ini_storage(self, origin_template):
        for temp in origin_template:
            if temp not in self.teme_storage:
                self.teme_storage[temp] = []
                self.best_pop[temp] = []
            self.teme_storage[temp].append(origin_template[temp])

    def sampling_temp(self, var_ope_index, code):
        operator_name = main_args["operators_to_optimize"][var_ope_index]
        raw = self.llm.draw_sample(
            self.Prompt.prompt_newTemplate(
                code,
                template_dict[operator_name + "_template"],
            )
        )
        text = raw.strip()
        if text.startswith("```"):
            nl = text.find("\n")
            text = text[nl + 1 :] if nl != -1 else text
            if text.strip().endswith("```"):
                text = text[: text.rfind("```")]
        template = text.strip()
        self.teme_storage[operator_name + "_template"].append(template)
        return template

    def update_best_ope(
        self, state, operators_best, scores, cycle_index, instance_path
    ):
        """save the best oprerators and return the best in all history"""
        pop = {"ope": operators_best, "scores": scores}
        self.pri_operators.append(pop)
        # 更新self.output_dir 路径下 ope_best_result.json，更新补充cycle_index的最佳记录和全局最佳记录
        best_ope_dict = max(self.pri_operators, key=lambda x: x["scores"])
        best_ope = best_ope_dict["ope"]
        # 准备当前周期的最佳记录
        current_best = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": cycle_index,
            "cycle_best_operators": operators_best,
            "score": scores,
            "operator": best_ope,  # 直接使用传入的操作符定义字典
        }
        # 确保输出目录存在
        json_file_path = os.path.join(
            "outputs", self.output_dir, f"best_results_{instance_path}.json"
        )
        save_operators_results(json_file_path, current_best)


class Node:
    def __init__(self, state, parent=None):
        self.state = state  # 当前状态，变量的组合
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0

    def ucb_score(self, c=1.41):
        # 计算UCB值，用于选择哪个子节点进行扩展
        if self.visit_count == 0:
            return float("inf")  # 未访问的节点优先扩展
        avg_reward = self.total_reward / self.visit_count
        return avg_reward + c * math.sqrt(
            math.log(self.parent.visit_count + 1) / self.visit_count
        )

    # 调试打印函数，打印当前节点的信息
    def print_node(self, level=0):
        indent = "  " * level  # 缩进，表示树的层级
        print(
            f"{indent}Node: {self.state}, Visit Count: {self.visit_count}, Reward: {self.total_reward}"
        )
        # 打印父节点
        if self.parent:
            print(f"{indent}  Parent: {self.parent.state}")
        else:
            print(f"{indent}  Parent: None")

        # 打印子节点
        for child in self.children:
            child.print_node(level + 1)


class MCTS:
    def __init__(
        self,
        main_args,
        llm,
        storages,
        max_iterations=1000,
        max_simulations=1,
        max_sampling_num=4,
    ):
        self.operators_to_optimize = main_args["operators_to_optimize"]
        self.main_args = main_args
        self.max_iterations = max_iterations
        self.max_simulations = max_simulations
        self.max_sampling_num = max_sampling_num
        self.llm = llm
        self.storages = storages  # 冷启动缓存
        self.output_dir = main_args["exp_dir"]
        [self.instance_path] = main_args["instance_paths"][14:15]

        self.problem = FJSP_Problem(obj_list, self.instance_path)

        self.history = {
            "best_solutions": [],
            "variable_spaces": [],
            "exploration_paths": [],
            "best_rewards": [],
        }
        self.history_var = {}

    def select(self, node):
        if not node.children:
            return node  # If no children, return the current node
        return max(node.children, key=lambda n: n.ucb_score())

    def _ini_state(self, node, origin_state):

        for next_var_index in range(len(origin_state)):
            var_ope = self.operators_to_optimize[next_var_index]
            if var_ope not in self.history_var:
                self.history_var[var_ope] = [origin_state[var_ope + "_template"]]

        # ini temaplte_storage
        self.temaplte_storage = temaplte_storage(
            self.main_args["template_dict"], self.llm
        )
        self.sampling_var_alltemplate()  # sampling new template

    def sampling_var_alltemplate(self):
        for var_ope_index in range(len(self.operators_to_optimize)):
            operation_name = self.operators_to_optimize[var_ope_index]
            alg_set = self.storages.latest[operation_name].individuals
            sorted_alg_set = sorted(alg_set, key=lambda x: (-x["score"], x["code"]))
            for num_index in range(self.max_sampling_num - 1):
                # 获取当前的最佳alg
                if len(
                    self.history_var[operation_name]
                ) <= self.max_sampling_num and num_index + 1 < len(sorted_alg_set):
                    alg_set = sorted_alg_set[num_index]
                    new_temp = self.temaplte_storage.sampling_temp(
                        var_ope_index, alg_set["code"]
                    )
                    self.history_var[operation_name].append(new_temp)

    def expand(self, generation, node):
        if len(node.state) >= 4:  # If the state is fully expanded
            return  # No more expansion needed

        next_var_index = len(node.state)  # Determine which variable to expand
        var_ope = self.operators_to_optimize[next_var_index]  # key is template name
        for value in self.history_var[var_ope]:
            new_state = node.state + [value]
            child_node = Node(new_state, parent=node)
            node.children.append(child_node)

    def simulate(self, generation, simuIndex, node):
        dir = str(generation) + "_" + str(simuIndex)
        # Simulate only if the state is incomplete (less than 4 variables)
        current_state = node.state
        while len(current_state) < 4:
            operator_name = self.operators_to_optimize[len(current_state)]
            next_value = random.choice(self.history_var[operator_name])
            current_state.append(next_value)
            # return 0
        operators_best, scores = quick_oc_eva(
            self.main_args, self.problem, dir, current_state
        )
        self.temaplte_storage.update_best_ope(
            current_state,
            operators_best,
            scores,
            dir,
            os.path.basename(self.instance_path),  # instance name
        )

        # After the state is complete, calculate the reward
        return scores

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def search(self):
        root = Node(state=[], parent=None)

        # 初始化默认取值和路径
        self._ini_state(root, self.main_args["template_dict"])

        # 每一代的记录
        for generation in range(self.max_iterations):
            node = root
            # 选择阶段
            while node.children:
                node = self.select(node)

            # 扩展阶段
            self.expand(generation, node)

            # 模拟阶段
            best_reward = float("inf")
            for simuIndex in range(self.max_simulations):
                reward = self.simulate(generation, simuIndex, node)
                if reward < best_reward:  # 只保留最优模拟结果
                    best_reward = reward

            # 回溯更新阶段
            self.backpropagate(node, best_reward)

            # 获取最优的完整解
            best_node = self.get_best_complete_node(root)

            if best_node is not None:
                # 如果找到了完整解
                self.history["best_solutions"].append(best_node.state)
                self.history["exploration_paths"].append(self.get_path(best_node))
                self.history["best_rewards"].append(best_reward)  # 记录每一代的最优奖励
            else:
                # 如果当前没有找到完整解，记录 None
                self.history["best_solutions"].append(None)
                self.history["exploration_paths"].append(None)
                self.history["best_rewards"].append(None)

            # 打印进度（可选）
            print(f"Generation {generation + 1}: Best Reward = {best_reward}")

        return self.history

    def save_results(self):
        exp_dir = self.output_dir
        # 保存到txt文件
        self.save_to_txt(exp_dir + "/mcts_history.txt")

        # 绘制收敛曲线并保存为 JPG 图片
        self.plot_convergence(exp_dir)

    def get_best_complete_node(self, root):
        best_node = None
        best_reward = float("-inf")

        for node in self.flatten_tree(root):
            if len(node.state) == 4:  # Complete state
                reward = (
                    node.total_reward / node.visit_count if node.visit_count > 0 else 0
                )
                if reward > best_reward:
                    best_reward = reward
                    best_node = node

        return best_node

    def flatten_tree(self, root):
        nodes = []
        nodes.append(root)
        for child in root.children:
            nodes.extend(self.flatten_tree(child))
        return nodes

    def get_path(self, node):
        path = []
        while node is not None:
            path.insert(0, node.state)
            node = node.parent
        return path

    def save_to_txt(self, filename="outputs/mcts_history.txt"):
        with open(filename, "w") as f:
            for i in range(len(self.history["best_solutions"])):
                f.write(f"Generation {i + 1}:\n")
                f.write(f"  Best Solution: {self.history['best_solutions'][i]}\n")
                f.write(f"  Variable Space: {self.history_var}\n")
                f.write(f"  Exploration Path: {self.history['exploration_paths'][i]}\n")
                f.write(f"  Best Reward: {self.history['best_rewards'][i]}\n\n")

    def plot_convergence(self, exp_dir):
        plt.plot(
            range(1, len(self.history["best_rewards"]) + 1),
            self.history["best_rewards"],
            label="Best Reward",
        )
        plt.xlabel("Generations")
        plt.ylabel("Best Reward")
        plt.title("Convergence Curve")
        plt.legend()

        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        plt.savefig(exp_dir + "/convergence_curve.jpg")
        plt.close()


if __name__ == "__main__":
    # main算法核心逻辑介绍：算子组合搜索，借助坐标轮换法，搜索最优算子组合
    ##############################
    operators_to_optimize = [
        "op_crossover",
        "op_mutation",
        "ma_crossover",
        "ma_mutation",
    ]  # need to be evolution operator

    template_dict = {
        "ma_crossover_template": ma_crossover_template,
        "ma_mutation_template": ma_mutation_template,
        "op_crossover_template": op_crossover_template,
        "op_mutation_template": op_mutation_template,
    }
    # 获取默认算子的源代码
    default_operators_source = {
        "op_crossover": inspect.getsource(op_crossover),
        "op_mutation": inspect.getsource(op_mutation),
        "ma_crossover": inspect.getsource(ma_crossover),
        "ma_mutation": inspect.getsource(ma_mutation),
    }
    # Eoh paras
    obj_list = ["makespan", "total_load"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = "MCT_FIX_fixing_stage3"  # fixed timestamp for result comparison
    exp_name = "FJSP_OC_TMAX"
    exp_dir = "FJSP_OC_" + timestamp
    log_dir = "outputs/" + exp_dir + "/eoh_log/{}"

    # 定义基准目录
    benchmark = "brandimarte"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(BASE_DIR, "Instances", benchmark)

    # 获取所有算例文件路径
    instance_paths = glob.glob(os.path.join(instances_dir, "*.txt"))
    instance_paths.sort()
    # print instance paths
    print(f"Found {len(instance_paths)} instances in '{benchmark}' benchmark:")
    for i, path in enumerate(instance_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    ##############################
    # initialize LLM
    llm = HttpsApi(None, None, None)
    llm._model = "Qwen3-30B-A3B"  # 改变了输入长度
    # llm.draw_sample("你是什么模型，什么版本，什么供应厂商的")
    # initialize templates

    max_cycles = 5  # CD 迭代次数
    main_args = {
        "n_evals": 3,  # 每次评估的重复次数 catl为6不需要调整
        "obj_list": ["makespan", "total_load"],
        "generation_num": 15,
        "pop_size": 100,  # 种群大小
        "eoh_max_generations": 5,
        "eoh_max_sample_nums": 25,  # max_sample_nums 大模型生成算子的个数 达到这个个数或者是迭代次数都会停止
        "operators_to_optimize": operators_to_optimize,
        "template_dict": template_dict,
        "default_operators_source": default_operators_source,
        "exp_dir": exp_dir,
        "instance_paths": instance_paths,
        "max_cycles": max_cycles,
        "eoh_pop_size": 5,  # 最小种群
        "eoh_num_samplers": 2,  # 单线程
        "eoh_num_evaluators": 2,  # 单线程
    }
    from _storage import Storages  # 极简 storages

    # 读取冷启动的信息
    storages = Storages(
        operators=operators_to_optimize,
        q_alpha=0.10,
        save_path=os.path.join("outputs", exp_dir, "storage", "storages.json"),
        ucb_beta=1.0,
    )
    # ====== warmup：有缓存就读；没缓存就跑一次并保存 ======
    benchmark = "brandimarte"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(BASE_DIR, "Instances", benchmark)
    instance_paths = glob.glob(os.path.join(instances_dir, "*.txt"))
    instance_paths.sort()
    if not instance_paths:
        raise FileNotFoundError(f"No instances found under {instances_dir}")
    target_idx = 14 if len(instance_paths) > 14 else 0
    instance_path = instance_paths[target_idx]
    instance_name = os.path.basename(instance_path)
    print(f"[INFO] Using single instance: {instance_name}")

    for operator_name in operators_to_optimize:
        cached = load_warmup_batch("MCT_FIX_fixing_stage2", operator_name)
        # 记录template
        storages.set_template(
            operator_name, cycle=-1, template=template_dict[operator_name + "_template"]
        )
        if cached is not None:
            # 回灌到 storages（注意：Storages 内部会把 -inf/NaN 视为无效→0，并统计 valid_rate）
            res = storages.record_batch(
                operator=operator_name, individuals=cached, cycle=0
            )
            continue

        problem = FJSP_Problem(obj_list, instance_path)

        # —— 无缓存：跑一次 EOH —— #
        template = template_dict[operator_name + "_template"]
        operator_evaluator = MyEvaluation(
            algorithm=NSGA_FJSP,
            problem=problem,
            exp_name=exp_name,
            template=template,
            ev_operator_name=operator_name,
            ref_point=problem.ref_points,
            generation_num=15,
            n_evals=5,
            pop_size=10,
        )
        operator_evaluator.save_operatorstr_bydict(default_operators_source)

        log_path = str(get_path(log_dir.format(f"warmup_{operator_name}")))
        manage_directory(log_path)
        profiler = TensorboardProfiler(
            wandb_project_name="aad-catl",
            log_dir=log_path,
            name=f"CATL@eoh@nsga_{operator_name}_warmup",
            create_random_path=False,
        )
        eoh = EoH(
            evaluation=operator_evaluator,
            profiler=profiler,
            llm=llm,
            debug_mode=True,
            max_generations=5,
            max_sample_nums=25,
            pop_size=5,
            num_samplers=3,
            num_evaluators=3,
        )
        eoh.run()

        # 序列化 & 存盘 & 回灌
        try:
            pop = getattr(
                eoh, "get_population", lambda: getattr(eoh, "_population", None)
            )()
        except Exception:
            pop = getattr(eoh, "_population", None)

        if not pop or not getattr(pop, "population", None):
            continue

        individuals = _serialize_population_simple(pop)
        save_warmup_batch(exp_dir, operator_name, individuals)

        res = storages.record_batch(
            operator=operator_name, individuals=individuals, cycle=0
        )

    # 创建MCTS对象
    mcts = MCTS(main_args, llm, storages, max_iterations=30, max_simulations=1)
    # 进行MCTS搜索并保存每一代的信息
    history = mcts.search()

    mcts.save_results()
