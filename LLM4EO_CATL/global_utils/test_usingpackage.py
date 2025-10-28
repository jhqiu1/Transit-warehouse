import random
import math
from copy import deepcopy


class OptimizationState:
    def __init__(self):
        # 初始化状态，四个变量初始值都为 0
        self.x1 = 0
        self.x2 = 0
        self.x3 = 0
        self.x4 = 0

    def getCurrentPlayer(self):
        # 轮流选择变量并返回当前的玩家
        if self.x1 == 0:
            return 1  # 轮到玩家 1 (最大化玩家) 选择 x1
        elif self.x2 == 0:
            return -1  # 轮到玩家 2 (最小化玩家) 选择 x2
        elif self.x3 == 0:
            return 1  # 轮到玩家 1 (最大化玩家) 选择 x3
        elif self.x4 == 0:
            return -1  # 轮到玩家 2 (最小化玩家) 选择 x4
        return None  # 所有变量都已选择

    def getPossibleActions(self):
        possibleActions = []

        # 根据变量状态返回可以选择的动作
        if self.x1 == 0:
            for i in range(1, 11):  # 假设 x1 的取值范围为 [1, 10]
                possibleActions.append(Action(player="x1", value=i))
        if self.x2 == 0:
            for i in range(5, 16):  # 假设 x2 的取值范围为 [5, 15]
                possibleActions.append(Action(player="x2", value=i))
        if self.x3 == 0:
            for i in range(10, 21):  # 假设 x3 的取值范围为 [10, 20]
                possibleActions.append(Action(player="x3", value=i))
        if self.x4 == 0:
            for i in range(20, 51):  # 假设 x4 的取值范围为 [20, 50]
                possibleActions.append(Action(player="x4", value=i))

        return possibleActions

    def takeAction(self, action):
        # 更新状态并返回新的状态
        newState = deepcopy(self)
        if action.player == "x1":
            newState.x1 = action.value
        elif action.player == "x2":
            newState.x2 = action.value
        elif action.player == "x3":
            newState.x3 = action.value
        elif action.player == "x4":
            newState.x4 = action.value
        return newState

    def isTerminal(self):
        # 终止条件：所有变量都已经选择
        return self.x1 != 0 and self.x2 != 0 and self.x3 != 0 and self.x4 != 0

    def getReward(self):
        # 目标函数：奖励为四个变量之和
        return self.x1 + self.x2 + self.x3 + self.x4


class Action:
    def __init__(self, player, value):
        self.player = player  # 动作对应的变量（x1, x2, x3, x4）
        self.value = value  # 该变量的取值

    def __str__(self):
        return f"{self.player} = {self.value}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.player == other.player
            and self.value == other.value
        )

    def __hash__(self):
        return hash((self.player, self.value))


import sys

sys.path.insert(0, sys.path[0] + "/../")
from MCTS.mcts import mcts

if __name__ == "__main__":
    # 初始化四变量优化状态
    initialState = OptimizationState()

    # 创建 MCTS 对象并设置时间限制（例如 1000 毫秒）
    searcher = mcts(timeLimit=1000)

    # 搜索并返回最优动作
    bestAction = searcher.search(initialState)

    print("Best Action: ", bestAction)
