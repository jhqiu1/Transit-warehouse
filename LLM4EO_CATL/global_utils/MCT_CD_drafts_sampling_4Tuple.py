import random
import math
import matplotlib.pyplot as plt
import os


# 假设目标函数是一个黑盒函数
def black_box_objective_function(variables):
    # 模拟黑盒目标函数，可以根据实际问题进行替换
    return sum(variables) + random.uniform(-5, 5)  # 添加一些噪声模拟黑盒不确定性


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


import random


class MCTS:
    def __init__(
        self, max_iterations=1000, max_simulations=10, max_sampling_num=4, max_depth=4
    ):
        self.max_iterations = max_iterations
        self.max_simulations = max_simulations
        self.max_sampling_num = max_sampling_num
        self.history = {
            "best_solutions": [],
            "variable_spaces": [],
            "exploration_paths": [],
            "best_rewards": [],
        }
        self.max_depth = max_depth
        self.history_var = {}

    def select(self, node):
        if not node.children:
            return node  # If no children, return the current node
        return max(node.children, key=lambda n: n.ucb_score())

    def _ini_state(self, node, origin_state):
        """初始化历史采样记录"""
        for next_var_index in range(len(origin_state)):
            if next_var_index not in self.history_var:
                self.history_var[next_var_index] = [origin_state[next_var_index]]
        # determine sampling generation
        start_gen = 1
        num_samples = self.max_sampling_num
        half_n = self.max_iterations // 2  # 前一半迭代的代次数
        # 计算采样间隔
        if half_n > start_gen:
            interval = (half_n - start_gen) // (num_samples - 1)
        else:
            interval = 1  # 如果 half_n 很小，则间隔设为1

        # 确定目标采样代次
        self.target_generations = [start_gen]
        for i in range(1, num_samples):
            target_gen = start_gen + i * interval
            # 确保最后一个采样点不超过 half_n
            if target_gen > half_n:
                target_gen = half_n
            self.target_generations.append(target_gen)

        # 凑够四元组
        while len(node.state) < 4:
            next_var_index = len(node.state)
            next_value = random.choice(self.history_var[next_var_index])
            node.state.append(next_value)

    def sampling_value(self, next_var_index):
        """从指定的分布中采样变量值"""
        if next_var_index == 0:
            # 假设第一个变量的范围为 [1, 10]
            min_val, max_val = 1, 10
            return random.uniform(min_val, max_val)  # 使用均匀分布进行采样

        elif next_var_index == 1:
            # 第二个变量的范围是 [5, 15]
            min_val, max_val = 5, 15
            return random.uniform(min_val, max_val)

        elif next_var_index == 2:
            # 第三个变量的范围是 [10, 20]
            min_val, max_val = 10, 20
            return random.uniform(min_val, max_val)

        elif next_var_index == 3:
            # 第四个变量的范围是 [20, 50]
            min_val, max_val = 20, 50
            return random.uniform(min_val, max_val)

    def samppling_vars(self, node):
        for var_ope_index in range(len(self.origin_state)):
            if len(self.history_var[var_ope_index]) <= self.max_sampling_num:
                self.history_var[var_ope_index].append(
                    self.sampling_value(var_ope_index)
                )

    def sampling_history_varbyone(self, next_var_index):
        return self.sampling_value(next_var_index)

    def expand(self, generation, node):
        if (
            generation <= 1 or len(self.get_path(node)) >= self.max_depth
        ):  # 判断当前node是否满足深度要求
            return  # No more expansion needed
        # 随机选取一个var进行template采样
        next_var_index = random.randint(0, len(self.origin_state) - 1)
        new_temp = self.sampling_history_varbyone(next_var_index)
        new_state = node.state
        new_state[next_var_index] = new_temp
        child_node = Node(new_state, parent=node)
        node.children.append(child_node)

    def simulate(self, node):
        """从当前节点开始进行随机模拟，确保当前变量都已赋值"""
        current_state = node.state

        # 计算奖励
        return black_box_objective_function(current_state)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def search(self, origin_state):
        self.origin_state = origin_state
        root = Node(state=[], parent=None)  # 从初始状态开始

        self._ini_state(root, origin_state)

        for generation in range(self.max_iterations):
            node = root
            while node.children:
                node = self.select(node)

            # 扩展阶段
            self.expand(generation, node)

            # 模拟阶段
            best_reward = float("inf")
            for _ in range(self.max_simulations):
                reward = self.simulate(node)
                if reward < best_reward:
                    best_reward = reward

            self.backpropagate(node, best_reward)

            # 获取最优的完整解
            best_node = self.get_best_complete_node(root)
            if best_node is not None:
                self.history["best_solutions"].append(best_node.state)
                self.history["exploration_paths"].append(self.get_path(best_node))
                self.history["best_rewards"].append(best_reward)

            # 打印进度
            print(f"Generation {generation + 1}: Best Reward = {best_reward}")

        return self.history

    def get_best_complete_node(self, root):
        best_node = None
        best_reward = float("-inf")

        for node in self.flatten_tree(root):
            if len(node.state) == 4:  # 完整状态
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

    def save_to_txt(self, filename="global_utils/outputs/mcts_history.txt"):
        with open(filename, "w") as f:
            for i in range(len(self.history["best_solutions"])):
                f.write(f"Generation {i + 1}:\n")
                f.write(f"  Best Solution: {self.history['best_solutions'][i]}\n")
                f.write(f"  Exploration Path: {self.history['exploration_paths'][i]}\n")
                f.write(f"  Best Reward: {self.history['best_rewards'][i]}\n\n")

    def plot_convergence(self):
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
        plt.savefig("global_utils/outputs/convergence_curve_4tuple.jpg")
        plt.close()


# 创建MCTS对象
mcts = MCTS(max_iterations=200, max_simulations=3)

origin_state = [1, 11, 21, 31]

# 进行MCTS搜索并保存每一代的信息
history = mcts.search(origin_state)

# 保存到txt文件
mcts.save_to_txt("global_utils/outputs/mcts_history_4tuple.txt")

# 绘制收敛曲线并保存为 JPG 图片
mcts.plot_convergence()
