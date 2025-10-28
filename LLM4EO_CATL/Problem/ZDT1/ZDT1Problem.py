import numpy as np

class ZDT1Problem:
    def __init__(self, n_variables=30):
        """
        ZDT1问题定义
        """
        self.n_variables = n_variables
        self.n_objectives = 2
        
    def evaluate(self, individual):
        """
        评估个体，返回两个目标函数值
        """
        # 目标函数1
        f1 = individual[0]
        
        # 计算g(x)
        g = 1 + 9 / (self.n_variables - 1) * np.sum(individual[1:])
        
        # 目标函数2
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        
        return np.array([f1, f2])
    
    def generate_individual(self):
        """
        生成随机个体
        """
        return np.random.random(self.n_variables)