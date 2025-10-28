import os
import glob
import time
import inspect
import random
from copy import deepcopy
from typing import Dict, Optional, Any, Tuple

from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH
from llm4ad.tools.profiler import TensorboardProfiler
from LLM.Prompts import Prompts
from _evaluate import (
    MyEvaluation,
    ma_crossover_template,
    ma_mutation_template,
    op_crossover_template,
    op_mutation_template,
    op_crossover,
    op_mutation,
    ma_crossover,
    ma_mutation,
)
from Problem.FJSP.FJSP_Problem import FJSP_Problem

from Algrithm.NSGA2 import NSGA_FJSP

# from Algrithm.MOEDA import MOEAD_FJSP
from logger.llm4eo_logger import ExperimentLogger
from utils.path_util import manage_directory
from _storage import Storages  # 极简 storages
from Agent.op_select_agent import (
    RandomEqualAllocator,
    UCBDecision,
    LLMDecisionPolicy,
    TemplateRewriteDecision,
    Window_UCBDecision,
    TemplateRuleDecision,  # according the rule to decise updating template
)
from utils.warmup_cache import (
    load_warmup_batch,
    _serialize_population_simple,
    save_warmup_batch,
)
from utils.time_setting import wait_until

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_path(*relative_parts):
    return os.path.join(PROJECT_ROOT, *relative_parts)


if __name__ == "__main__":
    # 代码定时器，到时间后执行
    # wait_until("04:00")
    # print("开始执行任务...")
    # ============ 基础参数 ============
    # NSGA para.
    n_evals = 5
    obj_list = ["makespan", "max_load"]
    generation_num = 15
    pop_size = 100

    # Eoh Para. normal
    # eoh_max_generations = 15  # 中等搜索深度
    # eoh_max_sample_nums = 200  # 充足评估预算
    # eoh_pop_size = 10  # 平衡多样性与深度
    # eoh_num_samplers = 2  # 并行采样
    # eoh_num_evaluators = 4  # 并行评估
    # quickly validation
    eoh_max_generations = 5  # 浅层搜索
    eoh_max_sample_nums = 25  # 最小评估次数
    eoh_pop_size = 5  # 最小种群
    eoh_num_samplers = 3  # 单线程
    eoh_num_evaluators = 3  # 单线程

    # 资源参数（注意：total_budget_samples 不包含 warmup 消耗）
    max_cycles = 20  # 决策轮次
    total_budget_samples = 100  # 主循环总采样预算（warmup 之外）
    # convergence_threshold = 0.01 # 提前终止条件

    operators_to_optimize = [
        "op_crossover",
        "op_mutation",
        "ma_crossover",
        "ma_mutation",
    ]

    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = "WinUCB_CTbyRule_QW8B"  # debug 限定不变
    exp_name = "FJSP_OC_TMAX"
    exp_dir = "FJSP_OC_TMAX_" + timestamp
    log_dir = "outputs/" + exp_dir + "/eoh_log/{}"

    # ============ 实例：只用一个 ============
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

    # ============ 初始化组件 ============
    llm = HttpsApi(None, None, None)
    Prompt = Prompts()

    template_dict = {
        "ma_crossover_template": ma_crossover_template,
        "ma_mutation_template": ma_mutation_template,
        "op_crossover_template": op_crossover_template,
        "op_mutation_template": op_mutation_template,
    }
    default_operators_source = {
        "op_crossover": inspect.getsource(op_crossover),
        "op_mutation": inspect.getsource(op_mutation),
        "ma_crossover": inspect.getsource(ma_crossover),
        "ma_mutation": inspect.getsource(ma_mutation),
    }

    # 存储（批次 + 历史）
    storages = Storages(
        operators=operators_to_optimize,
        q_alpha=0.10,
        save_path=os.path.join("outputs", exp_dir, "storage", "storages.json"),
        ucb_beta=1.0,
    )

    # random decision
    # Op_Policy = RandomEqualAllocator(
    #     operators=operators_to_optimize,
    #     max_cycles=max_cycles,
    #     total_budget_samples=total_budget_samples,
    #     seed=20250906,
    # )
    # UCB contro
    # Op_Policy = UCBDecision(
    #     operators=operators_to_optimize,
    #     c=1.0,
    #     seed=20250906,
    # )
    Op_Policy = Window_UCBDecision(
        operators=operators_to_optimize,
        c=1.0,
        seed=20250906,
    )
    # Op_Policy = LLMDecisionPolicy(
    #     llm=llm,
    #     operators=operators_to_optimize,
    # )
    Tem_Policy = TemplateRuleDecision(c=1.0, seed=42)

    # ============ 日志/问题/Evaluator ============
    logger = ExperimentLogger(exp_dir, instance_name)
    logger.log_info(f"===== Evaluating instance: {instance_name} =====")

    problem = FJSP_Problem(obj_list, instance_path)

    # 用于评估整套组合的 evaluator（后面复用）
    combo_evaluator = MyEvaluation(
        algorithm=NSGA_FJSP,
        problem=problem,
        exp_name=exp_name,
        template=None,
        ev_operator_name=None,
        ref_point=problem.ref_points,
        generation_num=generation_num,
        n_evals=n_evals,
        pop_size=pop_size,
    )

    # 当前/最佳算子代码字典（初始用默认）
    current_operators = deepcopy(default_operators_source)
    best_operators = deepcopy(current_operators)

    # =================================================================
    # ==== WARMUP：每个算子先采样  次，写入 Storages 作为特征初始化 ====
    # =================================================================
    # warmup_samples_per_op = 10
    logger.log_info(
        f"[Warmup] Each operator sampling {eoh_max_generations} times to initialize storage features"
    )

    # ====== warmup：有缓存就读；没缓存就跑一次并保存 ======
    for operator_name in operators_to_optimize:
        cached = load_warmup_batch("FJSP_OC_TMAX_UCB", operator_name)
        # 记录template
        storages.set_template(
            operator_name, cycle=-1, template=template_dict[operator_name + "_template"]
        )
        if cached is not None:
            # 回灌到 storages（注意：Storages 内部会把 -inf/NaN 视为无效→0，并统计 valid_rate）
            res = storages.record_batch(
                operator=operator_name, individuals=cached, cycle=0
            )
            logger.log_info(
                f"[Warmup][CacheHit] {operator_name} n={res['latest']['n_samples']}, "
                f"valid_rate={res['latest']['valid_rate']:.2%}, mean={res['latest']['mean_score']}"
            )
            continue
        continue

        # —— 无缓存：跑一次 EOH —— #
        template = template_dict[operator_name + "_template"]
        operator_evaluator = MyEvaluation(
            algorithm=NSGA_FJSP,
            problem=problem,
            exp_name=exp_name,
            template=template,
            ev_operator_name=operator_name,
            ref_point=problem.ref_points,
            generation_num=generation_num,
            n_evals=n_evals,
            pop_size=10,
        )
        operator_evaluator.save_operatorstr_bydict(current_operators)

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
            max_generations=eoh_max_generations,
            max_sample_nums=eoh_max_sample_nums,
            pop_size=eoh_pop_size,
            num_samplers=eoh_num_samplers,
            num_evaluators=eoh_num_evaluators,
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
            logger.log_warning(f"[Warmup] No population for {operator_name}, skip")
            continue

        individuals = _serialize_population_simple(pop)
        save_warmup_batch(exp_dir, operator_name, individuals)

        res = storages.record_batch(
            operator=operator_name, individuals=individuals, cycle=0
        )
        logger.log_info(
            f"[Warmup][CacheSave] {operator_name} n={res['latest']['n_samples']}, "
            f"valid_rate={res['latest']['valid_rate']:.2%}, mean={res['latest']['mean_score']}"
        )

    # ======= 在 warmup 之后，评估一次“当前默认组合”作为 baseline（用于后续比较） =======
    # best_score = combo_evaluator.evaluate_combination(current_operators)
    best_score = float("-inf")
    logger.log_info(
        f"[Baseline] Score after warmup (default operators): {best_score:.6f}"
    )
    # 如果你完全不想有 baseline，也可以把 best_score 设成 -inf，让第一轮必然接受改动

    # =================================================================
    # ==================== 主循环：由 决策模块 决定 =====================
    # =================================================================
    for cycle in range(max_cycles):
        logger.log_info(f"--- Decision cycle {cycle+1}/{max_cycles} ---")

        # 2) 决策：算子 + 本轮采样数
        op_features = storages.build_features()
        operator_name, do_rewrite = Op_Policy.decide(cycle, op_features)

        do_rewrite, _ = Tem_Policy.decide(
            cycle_idx=cycle, operator=operator_name, stor=storages
        )
        do_rewrite = False
        if do_rewrite:
            raw = llm.draw_sample(
                Prompt.prompt_newTemplate(
                    current_operators[operator_name],
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
            storages.set_template(operator_name, cycle=cycle, template=template)
            logger.log_info(
                f"[Rewrite Tem] operator={operator_name}, Template content:{template}"
            )

        else:
            template = template_dict[operator_name + "_template"]

        print(f"[Decision] pick operator={operator_name}, n_samples={generation_num}")

        # 4) 构造该算子的 evaluator
        operator_evaluator = MyEvaluation(
            algorithm=NSGA_FJSP,
            problem=problem,
            exp_name=exp_name,
            template=template,
            ev_operator_name=operator_name,
            ref_point=problem.ref_points,
            generation_num=generation_num,
            n_evals=n_evals,
            pop_size=pop_size,
        )
        operator_evaluator.save_operatorstr_bydict(current_operators)
        candidate_operators = deepcopy(current_operators)

        # 5) 运行 EOH
        log_path = str(get_path(log_dir.format(f"cycle{cycle+1}_{operator_name}")))
        manage_directory(log_path)
        profiler = TensorboardProfiler(
            wandb_project_name="aad-catl",
            log_dir=log_path,
            name=f"CATL@eoh@nsga_{operator_name}_cycle{cycle+1}",
            create_random_path=False,
        )
        eoh = EoH(
            evaluation=operator_evaluator,
            # max_sample_nums=int(n_samples),
            profiler=profiler,
            llm=llm,
            debug_mode=True,
            max_generations=eoh_max_generations,
            max_sample_nums=eoh_max_sample_nums,
            pop_size=eoh_pop_size,
            num_samplers=eoh_num_samplers,
            num_evaluators=eoh_num_evaluators,
        )
        eoh.run()

        # 6) 记录 population 到 storages
        try:
            ope_pops = getattr(
                eoh, "get_population", lambda: getattr(eoh, "_population", None)
            )()
        except Exception:
            ope_pops = getattr(eoh, "_population", None)

        if ope_pops and getattr(ope_pops, "population", None):
            res = storages.record_from_eoh(
                operator=operator_name,
                eoh_population=ope_pops,
                cycle=cycle + 1,
            )
            logger.log_info(
                f"[Storage] {ope_pops} batch: n={res['latest']['n_samples']}, "
                f"valid_rate={res['latest']['valid_rate']:.2%}, "
                f"best={res['latest']['best_score']}, mean={res['latest']['mean_score']}, "
                f"hist_valid_rate={res['history']['valid_rate']:.2%}, "
                f"hist_mean={res['history']['mean']}, mean_gain={res['diff']['mean_gain']}"
            )
        else:
            logger.log_warning(f"No valid population for {operator_name}; skip update")
            continue

        # 7) 从该批挑最优个体作为候选
        best_function = max(
            ope_pops.population, key=lambda ind: getattr(ind, "score", float("-inf"))
        )
        best_score_val = getattr(best_function, "score", None)
        if best_score_val is None or best_score_val == float("-inf"):
            logger.log_warning(f"Invalid best score for {operator_name}; skip update")
            continue

        candidate_operators[operator_name] = str(best_function)
        logger.log_info(
            f"[Candidate] {operator_name} local best score: {best_score_val:.6f}"
        )

        # 8) evaluate candidate_operators
        new_score = operator_evaluator.evaluate_combination(candidate_operators)
        logger.log_info(
            f"[Eval] After optimizing {operator_name}, score: {new_score:.6f}"
        )
        logger.record_iteration(
            cycle + 1, operator_name, candidate_operators, new_score
        )

        # 9) 贪婪接受
        if new_score > best_score:
            best_score = new_score
            current_operators = deepcopy(candidate_operators)
            best_operators = deepcopy(candidate_operators)
            logger.log_info(
                f"[Accept] New global best: {best_score:.6f} (by {operator_name})"
            )
            logger.record_best_result(
                cycle + 1, operator_name, best_operators, best_score
            )
        else:
            logger.log_info(f"[Reject] No improvement this cycle")

    # ============ record results ============
    logger.log_info(f"Optimization completed. Best score: {best_score}")
    logger.log_info("Best operator combination:")
    for op_name, op_source in best_operators.items():
        if isinstance(op_source, str):
            logger.log_info(f"  {op_name}: source code length = {len(op_source)}")
        else:
            logger.log_info(f"  {op_name}: default operator")

    # 保存最终结果（单实例）
    final_results = {
        instance_name: {
            "best_score": best_score,
            "best_operators": {
                k: v if isinstance(v, str) else "default"
                for k, v in best_operators.items()
            },
            "improvement_history": logger.best_results,
            "all_iterations": logger.all_iterations,
        }
    }
    logger.save_final_results(final_results)
