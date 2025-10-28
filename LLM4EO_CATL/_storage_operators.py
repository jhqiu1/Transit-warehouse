# _storages.py
from __future__ import annotations
import os, json, time, math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class Batch:
    """一次 eoh.run() 的整批信息（最近批次）"""

    operator: str
    cycle: Optional[int]
    step: Optional[int]
    ts: float
    n_samples: int
    # [{code:str, score:float, ...}]
    individuals: List[Dict[str, Any]] = field(default_factory=list)
    # 有效性
    valid_n: int = 0
    invalid_n: int = 0
    valid_rate: float = 0.0
    # 批次统计（基于“修正后分数”，即 -inf/NaN→0）
    best_score: Optional[float] = None
    worst_score: Optional[float] = None
    mean_score: Optional[float] = None
    median_score: Optional[float] = None
    std_score: Optional[float] = None
    q_low: Optional[float] = None
    q_high: Optional[float] = None
    best_idx: Optional[int] = None


def _stats(scores: np.ndarray, q_alpha: float = 0.10) -> Dict[str, Optional[float]]:
    """批次/历史的基础统计（scores 已为修正后的分数）"""
    if scores.size == 0:
        return dict(
            best=None,
            worst=None,
            mean=None,
            median=None,
            std=None,
            q_low=None,
            q_high=None,
            best_idx=None,
        )
    best_idx = int(np.argmax(scores))
    return dict(
        best=float(scores.max()),
        worst=float(scores.min()),
        mean=float(scores.mean()),
        median=float(np.median(scores)),
        std=float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        q_low=float(np.quantile(scores, q_alpha / 2.0)),
        q_high=float(np.quantile(scores, 1.0 - q_alpha / 2.0)),
        best_idx=best_idx,
    )


def _pct_gain(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """相对提升百分比 (new-old)/|old|，不可比时返回 None"""
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


class Storages:
    """
    极简按算子存储：
      - latest[op]: 最近一次 eoh.run() 的整批 + 统计（-inf/NaN 已修正为 0）
      - history_scores[op]: 历史所有个体“修正后分数”的扁平集合
      - 同时维护历史有效/总计数用于有效率
    """

    def __init__(
        self,
        operators: List[str],
        q_alpha: float = 0.10,
        save_path: Optional[str] = None,
        ucb_beta: float = 0.0,
    ):
        """
        q_alpha: 分位数区间（例如 0.10 -> 返回 5% 与 95% 分位）
        save_path: 若提供，每次记录后自动持久化
        ucb_beta: >0 时在 history() 里给出 UCB/LCB = mean ± beta*std/sqrt(n)
        """
        self.q_alpha = q_alpha
        self.save_path = save_path
        self.ucb_beta = float(ucb_beta)

        self.latest: Dict[str, Optional[Batch]] = {op: None for op in operators}
        self.history_scores: Dict[str, List[float]] = {op: [] for op in operators}
        self.history_counts: Dict[str, int] = {op: 0 for op in operators}  # total
        self.history_valid_counts: Dict[str, int] = {op: 0 for op in operators}  # valid

    # ---------------- 记录入口 ----------------

    def record_from_eoh(
        self,
        operator: str,
        eoh_population: Any,
        cycle: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        从 eoh population 对象记录新批次，并返回 {latest, history, diff}
        规则：
          - 个体分数 score == -inf 或 非有限（NaN/±inf） → 视为无效，存 0
          - latest 增加有效率；history 也维护有效率
        """
        pop = getattr(eoh_population, "population", None) or []
        individuals: List[Dict[str, Any]] = []
        for ind in pop:
            code = str(ind)
            raw = getattr(ind, "score", float("-inf"))
            try:
                sc = float(raw)
            except Exception:
                sc = float("-inf")
            valid = math.isfinite(sc) and sc != float("-inf")
            adj = sc if valid else 0.0
            individuals.append(
                {"code": code, "score": adj, "raw_score": sc, "valid": valid}
            )
        return self._finalize_record(operator, individuals, cycle, step)

    def record_batch(
        self,
        operator: str,
        individuals: List[Tuple[str, float]] | List[Dict[str, Any]],
        cycle: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """直接用 (code, score) 或 {"code":..,"score":..} 列表记录新批次（同样会做 -inf/NaN→0 修正与有效率统计）"""
        normalized: List[Dict[str, Any]] = []
        for it in individuals:
            if isinstance(it, dict):
                code = str(it.get("code", ""))
                raw = it.get("score", float("-inf"))
            else:
                code, raw = it  # type: ignore
                code, raw = str(code), raw
            try:
                sc = float(raw)
            except Exception:
                sc = float("-inf")
            valid = math.isfinite(sc) and sc != float("-inf")
            adj = sc if valid else 0.0
            normalized.append(
                {"code": code, "score": adj, "raw_score": sc, "valid": valid}
            )
        return self._finalize_record(operator, normalized, cycle, step)

    # ---------------- 查询接口 ----------------

    def latest_batch(self, operator: str) -> Optional[Dict[str, Any]]:
        b = self.latest.get(operator)
        return asdict(b) if b else None

    def history(self, operator: str) -> Dict[str, Any]:
        arr = np.array(self.history_scores.get(operator, []), dtype=float)
        s = _stats(arr, self.q_alpha)
        total = int(self.history_counts.get(operator, 0))
        valid = int(self.history_valid_counts.get(operator, 0))
        valid_rate = (valid / total) if total > 0 else None

        ucb = lcb = None
        if self.ucb_beta > 0 and s["mean"] is not None:
            denom = max(np.sqrt(arr.size), 1e-8)
            radius = self.ucb_beta * (s["std"] / denom if s["std"] is not None else 0.0)
            ucb = s["mean"] + radius
            lcb = s["mean"] - radius

        return {
            "operator": operator,
            "n_individuals": int(arr.size),
            "best": s["best"],
            "worst": s["worst"],
            "mean": s["mean"],
            "median": s["median"],
            "std": s["std"],
            "q_low": s["q_low"],
            "q_high": s["q_high"],
            "ucb": ucb,
            "lcb": lcb,
            "valid_count": valid,
            "total_count": total,
            "valid_rate": valid_rate,
        }

    def build_features(self) -> Dict[str, Dict[str, Any]]:
        feats = {}
        for op in self.latest.keys():
            feats[op] = {"latest": self.latest_batch(op), "history": self.history(op)}
        return feats

    # ---------------- 持久化 ----------------

    def dump(self, path: Optional[str] = None) -> None:
        path = path or self.save_path
        if not path:
            return
        data = {
            "q_alpha": self.q_alpha,
            "ucb_beta": self.ucb_beta,
            "history_scores": self.history_scores,
            "history_counts": self.history_counts,
            "history_valid_counts": self.history_valid_counts,
            "latest": {op: (asdict(b) if b else None) for op, b in self.latest.items()},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.q_alpha = float(data.get("q_alpha", self.q_alpha))
        self.ucb_beta = float(data.get("ucb_beta", self.ucb_beta))
        self.history_scores = {
            k: list(v) for k, v in data.get("history_scores", {}).items()
        }
        self.history_counts = {
            k: int(v) for k, v in data.get("history_counts", {}).items()
        }
        self.history_valid_counts = (
            {k: int(v) for k, v in data.get("history_valid_counts", {}).items()}
            if "history_valid_counts" in data
            else {k: 0 for k in self.history_counts.keys()}
        )
        raw_latest = data.get("latest", {})
        new_latest: Dict[str, Optional[Batch]] = {}
        for op, b in raw_latest.items():
            new_latest[op] = Batch(**b) if b else None
        self.latest = new_latest

    # ---------------- 内部：统一收尾 ----------------

    def _finalize_record(
        self,
        operator: str,
        individuals: List[Dict[str, Any]],
        cycle: Optional[int],
        step: Optional[int],
    ) -> Dict[str, Any]:
        # 分数已修正（score 字段）；同时有 raw_score 与 valid 标记
        scores = (
            np.array([x["score"] for x in individuals], dtype=float)
            if individuals
            else np.array([], dtype=float)
        )
        valid_n = int(sum(bool(x.get("valid", False)) for x in individuals))
        total_n = int(scores.size)
        invalid_n = total_n - valid_n
        valid_rate = (valid_n / total_n) if total_n > 0 else 0.0

        s = _stats(scores, self.q_alpha)
        batch = Batch(
            operator=operator,
            cycle=cycle,
            step=step,
            ts=time.time(),
            n_samples=total_n,
            individuals=individuals,
            valid_n=valid_n,
            invalid_n=invalid_n,
            valid_rate=valid_rate,
            best_score=s["best"],
            worst_score=s["worst"],
            mean_score=s["mean"],
            median_score=s["median"],
            std_score=s["std"],
            q_low=s["q_low"],
            q_high=s["q_high"],
            best_idx=s["best_idx"],
        )

        # 更新最近批次
        self.latest[operator] = batch
        # 扁平历史累计（使用“修正后分数”）
        if total_n > 0:
            self.history_scores[operator].extend(scores.tolist())
            self.history_counts[operator] += total_n
            self.history_valid_counts[operator] += valid_n

        # 历史统计（用于差异比较）
        hist = self.history(operator)
        diff = {
            "mean_gain": _pct_gain(batch.mean_score, hist["mean"]),
            "best_gain": _pct_gain(batch.best_score, hist["best"]),
            "median_gain": _pct_gain(batch.median_score, hist["median"]),
        }

        if self.save_path:
            self.dump(self.save_path)

        return {"latest": asdict(batch), "history": hist, "diff": diff}
