import os
import json
import logging
import hashlib
from datetime import datetime


def _to_jsonable(obj):
    """尽量转成可 JSON 序列化形态。"""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _sha1(s):
    if not isinstance(s, str):
        return None
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def _preview(s, n):
    if not isinstance(s, str):
        return None
    return s if len(s) <= n else (s[:n] + "... [trunc]")


def _summarize_ops(ops_dict: dict) -> dict:
    """只记录源码长度和sha1，避免把整段代码写进日志。"""
    out = {}
    for k, v in (ops_dict or {}).items():
        if isinstance(v, str):
            out[k] = {"len": len(v), "sha1": _sha1(v)}
        else:
            out[k] = {"type": type(v).__name__}
    return out


class ExperimentLogger:
    """实验日志记录器，用于记录实验过程和结果"""

    def __init__(self, exp_dir, instance_name):
        self.exp_dir = exp_dir
        self.instance_name = instance_name
        self.log_dir = os.path.join("outputs", exp_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{instance_name}_{timestamp}.log")

        # 独立 logger，避免污染 root
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        self.logger = logging.getLogger(f"exp.{self.instance_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        # 路径
        self.iteration_file = os.path.join(
            self.log_dir, f"iterations_{self.instance_name}.json"
        )
        self.best_result_file = os.path.join(
            "outputs", self.exp_dir, f"best_results_{self.instance_name}.json"
        )
        self.decision_log_path = os.path.join(self.log_dir, "decision_log.jsonl")
        self.final_result_file = os.path.join(
            "outputs", self.exp_dir, "all_instances_results.json"
        )

        # 内存中的累积数据
        self.all_iterations = []
        self.best_results = []
        self.decision_events = (
            []
        )  # <—— 新增：把每次决策也存到内存，最终合并到 final_results
        self.iteration_logl = os.path.join(self.log_dir, "iterations.jsonl")
        self.best_logl = os.path.join(self.log_dir, "best_results.jsonl")

    # 基本日志
    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    # 迭代与最佳
    def record_iteration(self, cycle, operator_name, operators, score):
        iteration_data = {
            "timestamp": datetime.now().isoformat(),
            "cycle": cycle,
            "operator": operator_name,
            "score": score,
            "operators": {
                k: v if isinstance(v, str) else "default" for k, v in operators.items()
            },
        }
        self.all_iterations.append(iteration_data)
        # 1) 逐条追加 JSONL（增量写、crash 也不丢）
        with open(self.iteration_logl, "a", encoding="utf-8") as f:
            f.write(json.dumps(iteration_data, ensure_ascii=False) + "\n")
            f.flush()  # 可选：更实时
            # import os; os.fsync(f.fileno())  # 可选：更抗崩溃

        # 2) 同时更新快照 JSON（可留可去）
        with open(self.iteration_file, "w", encoding="utf-8") as f:
            json.dump(self.all_iterations, f, indent=4, ensure_ascii=False)

    def record_best_result(self, cycle, operator_name, operators, score):
        best_result = {
            "timestamp": datetime.now().isoformat(),
            "cycle": cycle,
            "operator": operator_name,
            "score": score,
            "operators": {
                k: v if isinstance(v, str) else "default" for k, v in operators.items()
            },
        }
        self.best_results.append(best_result)
        os.makedirs(os.path.dirname(self.best_result_file), exist_ok=True)
        with open(self.best_result_file, "w", encoding="utf-8") as f:
            json.dump(self.best_results, f, indent=4, ensure_ascii=False)
        return best_result

    # 决策事件（新增）
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
        evt = {
            "ts": datetime.now().isoformat(),
            "cycle": int(cycle) + 1,  # 1-based 更易读
            "operator": operator_name,
            "do_rewrite": bool(do_rewrite),
            "template_preview": _preview(template, 400),
            "template_sha1": _sha1(template),
            "accepted": bool(accepted),
            "scores": {
                "before": float(best_score_before),
                "candidate": float(new_score),
                "global_best_after": float(best_score_after),
            },
            "op_features": _to_jsonable(op_features),
            "current_ops_summary": _summarize_ops(current_ops),
            "candidate_ops_summary": _summarize_ops(candidate_ops),
        }
        if extra:
            evt["extra"] = _to_jsonable(extra)

        # 1) 逐行写 JSONL（便于追溯）
        with open(self.decision_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        # 2) 内存也存一份（便于最终合并）
        self.decision_events.append(evt)

        # 控制台摘要
        self.log_info(
            f"[Decision] cycle={evt['cycle']} op={operator_name} "
            f"rewrite={do_rewrite} accepted={accepted} "
            f"score_before={best_score_before:.6f} "
            f"candidate={new_score:.6f} best_after={best_score_after:.6f}"
        )

    # 最终结果
    def save_final_results(self, all_results: dict):
        """
        把传入的 all_results 与迭代/最佳/决策事件合并保存。
        额外写入 artifacts，标注各类输出文件的路径。
        """
        payload = {
            "results": _to_jsonable(all_results) if all_results is not None else {},
            "iterations": self.all_iterations,
            "best_results": self.best_results,
            "decisions": self.decision_events,  # <—— 决策事件也并入最终 JSON
            "artifacts": {
                "logs_dir": self.log_dir,
                "run_log": self.log_file,
                "decision_log_jsonl": self.decision_log_path,
                "iterations_json": self.iteration_file,
                "best_results_json": self.best_result_file,
            },
            "saved_at": datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(self.final_result_file), exist_ok=True)
        with open(self.final_result_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        self.log_info(f"All instances results saved to: {self.final_result_file}")
