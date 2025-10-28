import numpy as np
from Logger import GALogger, get_logger
import time
import Problem.ZDT1.ZDT1Problem as ZDT1Problem
import Algrithm.NSGA2 as NSGA2
import matplotlib.pyplot as plt
import json

# 主函数
if __name__ == "__main__":
    # 配置日志系统
    tracked_metrics = [
        "f1_min", "f1_max", "f1_avg",
        "f2_min", "f2_max", "f2_avg",
        "selection_time", "crossover_time", "mutation_time", "evaluation_time", "total_time"
    ]
    
    # 初始化日志管理器
    logger = GALogger(
        exp_name="NSGA2_ZDT1",
        tracked_metrics=tracked_metrics
    )
    
    # 记录实验配置
    config = {
        "problem": "ZDT1",
        "n_variables": 30,
        "population_size": 100,
        "max_generations": 50,
        "crossover_prob": 0.9,
        "mutation_prob": 0.1
    }
    logger.log_config(config)
    
    # 创建问题实例
    problem = ZDT1Problem.ZDT1Problem(n_variables=config["n_variables"])
    
    # 创建NSGA-II算法实例
    nsga2 = NSGA2.NSGA2(
        problem,
        population_size=config["population_size"],
        max_generations=config["max_generations"],
        crossover_prob=config["crossover_prob"],
        mutation_prob=config["mutation_prob"]
    )
    
    # 运行算法
    start_time = time.time()
    population, fitness = nsga2.run()
    total_time = time.time() - start_time
    
    # 记录总时间
    logger.experiment_data["config"]["total_time"] = total_time
    logger.logger.info(f"总运行时间: {total_time:.2f}秒")
    
    # 保存最终帕累托前沿
    f1_values = [f[0] for f in fitness]
    f2_values = [f[1] for f in fitness]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(f1_values, f2_values, alpha=0.5)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"Final Pareto Front")
    plt.grid(True)
    
    # 保存图像
    plot_file = f"{logger.plot_dir}/final_pareto_front.png"
    plt.savefig(plot_file)
    plt.close()
    logger.logger.info(f"最终帕累托前沿已保存: {plot_file}")
    
    # 保存最优解
    best_solutions = []
    for i, f in enumerate(fitness):
        best_solutions.append({
            "individual": population[i].tolist(),
            "fitness": f.tolist()
        })
    
    solution_file = f"{logger.data_dir}/best_solutions.json"
    with open(solution_file, "w") as f:
        json.dump(best_solutions, f, indent=2)
    logger.logger.info(f"最优解已保存: {solution_file}")
    
    # 保存结果
    logger.save_results()
    
    print("实验完成! 所有结果保存在:", logger.output_dir)