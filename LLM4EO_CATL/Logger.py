import os
import time
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import sys


def _sha1(s):
    if not isinstance(s, str):
        return None
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def _preview(s, n):
    if not isinstance(s, str):
        return None
    return s if len(s) <= n else (s[:n] + "... [trunc]")


def _to_jsonable(obj):
    """尽量把对象变成可 JSON 序列化的形态（dict/list/标量）"""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    # 兜底：字符串化
    return str(obj)


def _summarize_ops(ops_dict: dict) -> dict:
    """只记录源码长度和sha1，避免把整段代码写进日志"""
    out = {}
    for k, v in (ops_dict or {}).items():
        if isinstance(v, str):
            out[k] = {"len": len(v), "sha1": _sha1(v)}
        else:
            out[k] = {"type": type(v).__name__}
    return out


class Logger:
    def __init__(self, exp_name="FJSP_Experiment", output_dir="evaluation_logs"):
        """
        初始化日志记录器

        参数:
            exp_name: 实验名称
            output_dir: 输出目录
        """
        # 创建时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 创建顶层实验目录
        self.top_level_dir = os.path.join(output_dir, f"{exp_name}_{timestamp}")
        os.makedirs(self.top_level_dir, exist_ok=True)

        # 初始化数据结构
        self.data = {
            "config": {
                "experiment_name": exp_name,
                "start_time": timestamp,
                "top_level_dir": self.top_level_dir,
            },
            "results": defaultdict(list),
            "pareto_fronts": {},
            "operator_performance": defaultdict(lambda: defaultdict(list)),
        }

        # 决策事件日志（JSONL）
        self.decision_events = []
        self.decision_log_path = os.path.join(self.top_level_dir, "decision_log.jsonl")

        # 初始化日志系统 & 重定向 stdout
        self._init_logging()
        self._redirect_stdout()

        self.logger.info(f"Experiment '{exp_name}' started at {time.ctime()}")
        self.logger.info(f"All outputs will be saved to: {self.top_level_dir}")

    def _init_logging(self):
        """初始化日志系统"""
        # 创建日志记录器
        self.logger = logging.getLogger("FJSPLogger")
        self.logger.setLevel(logging.DEBUG)

        # 文件日志
        log_file = os.path.join(self.top_level_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        file_handler.setFormatter(file_format)

        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # 清理重复 handler（防止多次实例化重复输出）
        if self.logger.handlers:
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _redirect_stdout(self):
        """重定向标准输出到文件"""
        # 创建输出文件
        self.stdout_file = open(
            os.path.join(self.top_level_dir, "stdout.log"), "w", encoding="utf-8"
        )

        # 保存原始标准输出
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # 重定向标准输出
        sys.stdout = self.stdout_file
        sys.stderr = self.stdout_file

    def log_info(self, message):
        """记录信息级别日志"""
        self.logger.info(message)

    def log_config(self, config):
        """记录实验配置"""
        self.data["config"].update(config)
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_operator(self, operator_name, operator_config):
        """记录算子配置"""
        if "operators" not in self.data["config"]:
            self.data["config"]["operators"] = {}

        self.data["config"]["operators"][operator_name] = operator_config
        self.logger.info(f"Operator '{operator_name}' configuration logged")

    def log_result(self, run_id, result):
        """记录单次运行结果"""
        self.data["results"][run_id].append(result)
        self.logger.debug(f"Result for run {run_id} logged")

    def log_pareto_front(self, operator_name, pareto_front):
        """记录帕累托前沿"""
        self.data["pareto_fronts"][operator_name] = pareto_front.tolist()
        self.logger.debug(f"Pareto front for operator '{operator_name}' logged")

    def log_run_start(self, run_id, total_runs):
        """记录运行开始"""
        self.logger.info(f"===== Starting run {run_id+1}/{total_runs} =====")

    def log_run_complete(self, run_id, total_runs, hv, pareto_front):
        """记录运行完成并显示详细指标"""
        num_solutions = len(pareto_front)
        avg_makespan = np.mean(pareto_front[:, 0])
        avg_utilization = np.mean(pareto_front[:, 1])
        min_makespan = np.min(pareto_front[:, 0])
        max_utilization = np.max(pareto_front[:, 1])

        self.logger.info(f"Run {run_id+1}/{total_runs} completed")
        self.logger.info(f"  HV: {hv:.4f}")
        self.logger.info(f"  Pareto solutions: {num_solutions}")
        self.logger.info(f"  Makespan: Avg={avg_makespan:.2f}, Min={min_makespan:.2f}")
        self.logger.info(
            f"  Utilization: Avg={avg_utilization:.4f}, Max={max_utilization:.4f}"
        )

    # =========================
    # 新增：决策事件记录方法
    # =========================
    def log_decision(
        self,
        *,
        cycle: int,
        op_features: dict,
        operator_name: str,
        do_rewrite: bool,
        template: str,
        accepted: bool,
        current_ops: dict,
        candidate_ops: dict,
        new_score: float,
        best_score_before: float,
        best_score_after: float,
        extra: dict,
    ) -> None:
        """
        记录“每次决策”的完整信息，落盘到 JSONL，并打印简要摘要。

        参数：
          - cycle: 从 0 开始的循环索引（会存为 1-based）
          - op_features: storages.build_features() 的返回
          - operator_name: 本轮选择优化的算子名
          - do_rewrite: 是否进行模板重写
          - template: 本轮使用的模板文本（可能是默认/重写）
          - accepted: 本轮候选是否被接受（是否更新算子组合）
          - current_ops: 更新前的算子组合（dict: name -> 源码字符串）
          - candidate_ops: 将本轮候选替换后的组合（dict）
          - new_score: 该候选组合的得分
          - best_score_before: 更新前的 best_score
          - best_score_after: 更新后的 best_score（若拒绝则与 before 相同）
          - extra: 可选的附加指标（如 mean_gain / valid_rate 等）
        """
        evt = {
            "ts": time.time(),
            "cycle": cycle + 1,  # 1-based
            "operator": operator_name,
            "do_rewrite": bool(do_rewrite),
            "template_preview": _preview(template, 400),
            "template_sha1": _sha1(template),
            "scores": {
                "before": best_score_before,
                "candidate": new_score,
                "global_best_after": best_score_after,
            },
            "accepted": bool(accepted),
            "op_features": _to_jsonable(op_features),
            "current_ops_summary": _summarize_ops(current_ops),
            "candidate_ops_summary": _summarize_ops(candidate_ops),
        }
        if extra:
            evt["extra"] = _to_jsonable(extra)

        # 追加写入 JSONL
        with open(self.decision_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

        self.decision_events.append(evt)

        # 控制台简要摘要
        self.log_info(
            f"[DecisionLog] cycle={evt['cycle']} op={operator_name} "
            f"rewrite={do_rewrite} accepted={accepted} "
            f"score_before={best_score_before:.6f} score_candidate={new_score:.6f} "
            f"best_after={best_score_after:.6f}"
        )

    def save_results(self):
        """保存所有结果并生成图表"""
        # 添加结束时间
        self.data["config"]["end_time"] = time.strftime("%Y%m%d_%H%M%S")

        # 保存JSON数据
        results_file = os.path.join(self.top_level_dir, "results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        # 生成图表
        self._generate_plots()

        # 恢复标准输出
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.stdout_file.close()

        self.logger.info(
            f"Experiment completed! All results saved to: {self.top_level_dir}"
        )

    def _generate_plots(self):
        """生成核心图表"""
        # 创建图表目录
        plots_dir = os.path.join(self.top_level_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # HV值比较图
        if self.data["results"]:
            plt.figure(figsize=(10, 6))
            hvs = [
                r["hv"] for results in self.data["results"].values() for r in results
            ]
            if hvs:
                plt.bar(range(len(hvs)), hvs)
                plt.xlabel("Run ID")
                plt.ylabel("Hypervolume (HV)")
                plt.title("HV Comparison")
                plot_file = os.path.join(plots_dir, "hv_comparison.png")
                plt.savefig(plot_file)
                plt.close()
                self.logger.info(f"HV comparison plot saved: {plot_file}")

        # 帕累托前沿比较图
        if self.data["pareto_fronts"]:
            plt.figure(figsize=(10, 8))
            for operator, front in self.data["pareto_fronts"].items():
                points = np.array(front)
                if len(points) > 0:
                    sorted_indices = np.argsort(points[:, 0])
                    plt.plot(
                        points[sorted_indices, 0],
                        points[sorted_indices, 1],
                        "o-",
                        label=operator,
                    )

            plt.xlabel("Makespan")
            plt.ylabel("Utilization")
            plt.title("Pareto Front Comparison")
            plt.legend()
            plot_file = os.path.join(plots_dir, "pareto_comparison.png")
            plt.savefig(plot_file)
            plt.close()
            self.logger.info(f"Pareto front comparison plot saved: {plot_file}")

        # 目标性能图
        if self.data["results"]:
            plt.figure(figsize=(12, 8))

            # 完工时间
            plt.subplot(2, 1, 1)
            makespans = [
                r["makespan"]
                for results in self.data["results"].values()
                for r in results
            ]
            if makespans:
                plt.plot(range(len(makespans)), makespans, "b-", label="Makespan")
                plt.xlabel("Run ID")
                plt.ylabel("Makespan")
                plt.title("Makespan Performance")
                plt.grid(True)

            # 设备利用率
            plt.subplot(2, 1, 2)
            utilizations = [
                r["utilization"]
                for results in self.data["results"].values()
                for r in results
            ]
            if utilizations:
                plt.plot(
                    range(len(utilizations)), utilizations, "r-", label="Utilization"
                )
                plt.xlabel("Run ID")
                plt.ylabel("Utilization")
                plt.title("Utilization Performance")
                plt.grid(True)

            plt.tight_layout()
            plot_file = os.path.join(plots_dir, "objective_performance.png")
            plt.savefig(plot_file)
            plt.close()
            self.logger.info(f"Objective performance plot saved: {plot_file}")
