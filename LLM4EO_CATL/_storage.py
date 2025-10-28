# _storages.py
from __future__ import annotations
import os, json, time, math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


# ---------------- 基础数据结构 ----------------


@dataclass
class Batch:
    """一次 eoh.run() 的整批信息（最近批次）"""

    operator: str
    cycle: Optional[int]
    ts: float
    n_samples: int
    # 本批次归属的模板名（来自“当前模板”或 set_template 设置）
    template: Optional[str] = None

    # [{code:str, score:float, raw_score:float, valid:bool}]
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


# ---------------- 核心存储器 ----------------


class Storages:
    """
    极简按算子存储（保持原统计不变），并新增“当前模板 & 历史模板集合”：
      - latest[op]: 最近一次 eoh.run() 的整批（-inf/NaN 已修正为 0）
      - history_scores[op]: 历史所有个体“修正后分数”的扁平集合
      - 有效率统计仍按算子聚合

    新增：
      - current_template[op]: 当前模板名（由 set_template() 更新，或首次记录时绑定）
      - templates_seen[op]: 历史用过的模板列表（去重，按出现顺序）
      - latest_by_template[op][tmpl]: 该算子在此模板下的最近一次 Batch（仅便于查询）
      - template_use_counts[op][tmpl]: 该模板被记录过的批次数
      - template_change_log[op]: 模板变更日志（[{cycle, ts, template}]）
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

        # 原有：算子聚合
        self.latest: Dict[str, Optional[Batch]] = {op: None for op in operators}
        self.history_scores: Dict[str, List[float]] = {op: [] for op in operators}
        self.history_counts: Dict[str, int] = {op: 0 for op in operators}  # total
        self.history_valid_counts: Dict[str, int] = {op: 0 for op in operators}  # valid

        # 新增：模板状态与便捷查询
        self.current_template: Dict[str, Optional[str]] = {op: None for op in operators}
        self.templates_seen: Dict[str, List[str]] = {op: [] for op in operators}
        self.latest_by_template: Dict[str, Dict[str, Optional[Batch]]] = {
            op: {} for op in operators
        }
        self.template_use_counts: Dict[str, Dict[str, int]] = {
            op: {} for op in operators
        }
        self.template_change_log: Dict[str, List[Dict[str, Any]]] = {
            op: [] for op in operators
        }

    # ---------------- 模板管理 ----------------

    def set_template(
        self, operator: str, cycle: Optional[int], template: Optional[str]
    ) -> None:
        """
        外部每次更新模板后调用：
          - 储存当前算子、轮次、模板（写入变更日志）
          - 设置为当前模板，并更新历史模板集合
        """
        self._ensure_op(operator)
        self.current_template[operator] = template
        ts = time.time()
        self.template_change_log[operator].append(
            {"cycle": cycle, "ts": ts, "template": template}
        )
        if template and template not in self.templates_seen[operator]:
            self.templates_seen[operator].append(template)

    def get_current_template(self, operator: str) -> Optional[str]:
        self._ensure_op(operator)
        return self.current_template.get(operator)

    def get_templates(self, operator: str) -> List[str]:
        self._ensure_op(operator)
        return list(self.templates_seen.get(operator, []))

    def get_template_change_log(self, operator: str) -> List[Dict[str, Any]]:
        self._ensure_op(operator)
        return list(self.template_change_log.get(operator, []))

    # ---------------- 记录入口（不传模板，默认当前模板） ----------------

    def record_from_eoh(
        self,
        operator: str,
        eoh_population: Any,
        cycle: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        从 eoh population 对象记录新批次，并返回 {latest, history, diff}
        规则：
          - 个体分数 score == -inf 或 非有限（NaN/±inf） → 视为无效，存 0
          - latest 增加有效率；history 也维护有效率
          - 模板：自动使用 current_template[operator] 标注
        """
        self._ensure_op(operator)
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
        return self._finalize_record(operator, individuals, cycle)

    def record_batch(
        self,
        operator: str,
        individuals: List[Tuple[str, float]] | List[Dict[str, Any]],
        cycle: Optional[int] = None,
    ) -> Dict[str, Any]:
        """直接用 (code, score) 或 {"code":..,"score":..} 列表记录新批次（同样会做 -inf/NaN→0 修正与有效率统计）"""
        self._ensure_op(operator)
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
        return self._finalize_record(operator, normalized, cycle)

    # ---------------- 查询接口（原聚合 + 模板便捷） ----------------

    def latest_batch(self, operator: str) -> Optional[Dict[str, Any]]:
        """算子层面的最近一次（不分模板聚合）"""
        b = self.latest.get(operator)
        return asdict(b) if b else None

    def history(self, operator: str) -> Dict[str, Any]:
        """算子层面的聚合统计（保持原逻辑）"""
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

    # ---- 新增：满足你 3/4 点需求的查询 ----

    def get_current_template_features(self, operator: str) -> Dict[str, Any]:
        """
        查询“算子当前模板对应”的特征信息
          - current_template
          - latest_of_current_template: 当前模板下最近一次批次（若无则 None）
          - template_meta: {use_count, last_used_ts, last_used_cycle}
        """
        self._ensure_op(operator)
        tmpl = self.current_template.get(operator)
        latest_tmpl_batch = None
        meta = None
        if tmpl:
            b = self.latest_by_template[operator].get(tmpl)
            latest_tmpl_batch = asdict(b) if b else None
            use_count = self.template_use_counts[operator].get(tmpl, 0)
            # last_used_ts / cycle 取自 b（如果有）
            last_ts = b.ts if b else None
            last_cycle = b.cycle if b else None
            meta = {
                "use_count": use_count,
                "last_used_ts": last_ts,
                "last_used_cycle": last_cycle,
            }

        return {
            "operator": operator,
            "current_template": tmpl,
            "latest_of_current_template": latest_tmpl_batch,
            "template_meta": meta,
        }

    def get_all_templates_features(self, operator: str) -> Dict[str, Dict[str, Any]]:
        """
        查询“历史所有模板的特征信息”（仅做元信息与最近一次批次，统计仍按算子聚合）
          返回 {template: {"latest_batch":..., "use_count":..., "last_used_ts":..., "last_used_cycle":...}}
        """
        self._ensure_op(operator)
        out: Dict[str, Dict[str, Any]] = {}
        for tmpl in self.templates_seen.get(operator, []):
            b = self.latest_by_template[operator].get(tmpl)
            out[tmpl] = {
                "latest_batch": (asdict(b) if b else None),
                "use_count": self.template_use_counts[operator].get(tmpl, 0),
                "last_used_ts": (b.ts if b else None),
                "last_used_cycle": (b.cycle if b else None),
            }
        return out

    def get_operator_features(self, operator: str) -> Dict[str, Any]:
        """
        查询“当前算子所有的特征信息”（你第4点）：
          - current_template / templates_seen
          - latest（算子层面最近一次）
          - history（算子聚合统计）
          - by_template_meta（每个模板的 latest + meta 速览）
        """
        self._ensure_op(operator)
        latest = self.latest_batch(operator)
        hist = self.history(operator)
        by_t = self.get_all_templates_features(operator)
        return {
            "operator": operator,
            "current_template": self.current_template.get(operator),
            "templates_seen": self.templates_seen.get(operator, []),
            "latest": latest,
            "history": hist,
            "by_template_meta": by_t,
        }

    def build_features(self) -> Dict[str, Dict[str, Any]]:
        """保持原接口，顺便暴露当前模板与模板集合"""
        feats = {}
        for op in self.latest.keys():
            feats[op] = {
                "current_template": self.current_template.get(op),
                "templates_seen": self.templates_seen.get(op, []),
                "latest": self.latest_batch(op),
                "history": self.history(op),
            }
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
            # 新增字段
            "current_template": self.current_template,
            "templates_seen": self.templates_seen,
            "latest_by_template": {
                op: {t: (asdict(b) if b else None) for t, b in tb.items()}
                for op, tb in self.latest_by_template.items()
            },
            "template_use_counts": self.template_use_counts,
            "template_change_log": self.template_change_log,
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

        # 原有：算子聚合
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

        # latest（兼容旧格式：可能包含 step 或缺少 template）
        raw_latest = data.get("latest", {})
        new_latest: Dict[str, Optional[Batch]] = {}
        for op, b in raw_latest.items():
            if b:
                b = dict(b)
                # 兼容：老记录若有 step，忽略；若无 template，补 None
                if "step" in b:
                    b.pop("step", None)
                b.setdefault("template", None)
                new_latest[op] = Batch(**b)
            else:
                new_latest[op] = None
        self.latest = new_latest

        # 新增字段（兼容：若不存在则初始化）
        self.current_template = {
            k: v for k, v in data.get("current_template", {}).items()
        }
        self.templates_seen = {
            k: list(v) for k, v in data.get("templates_seen", {}).items()
        }

        # latest_by_template（兼容：若不存在则为空）
        self.latest_by_template = {}
        for op, tb in data.get("latest_by_template", {}).items():
            self.latest_by_template[op] = {}
            for t, b in (tb or {}).items():
                if b:
                    b = dict(b)
                    if "step" in b:
                        b.pop("step", None)
                    b.setdefault(
                        "template",
                        t if b.get("template") is None else b.get("template"),
                    )
                    self.latest_by_template[op][t] = Batch(**b)
                else:
                    self.latest_by_template[op][t] = None

        self.template_use_counts = {
            op: {t: int(c) for t, c in (mp or {}).items()}
            for op, mp in data.get("template_use_counts", {}).items()
        }
        self.template_change_log = {
            op: list(v) for op, v in data.get("template_change_log", {}).items()
        }

        # 补齐缺省键
        for op in set(list(self.latest.keys()) + list(self.history_scores.keys())):
            self._ensure_op(op)

    # ---------------- 内部：统一收尾 ----------------

    def _finalize_record(
        self,
        operator: str,
        individuals: List[Dict[str, Any]],
        cycle: Optional[int],
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
        tmpl = self.current_template.get(operator)

        batch = Batch(
            operator=operator,
            cycle=cycle,
            ts=time.time(),
            n_samples=total_n,
            template=tmpl,
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

        # 更新最近批次（算子层面）
        self.latest[operator] = batch
        # 扁平历史累计（使用“修正后分数”）
        if total_n > 0:
            self.history_scores[operator].extend(scores.tolist())
            self.history_counts[operator] += total_n
            self.history_valid_counts[operator] += valid_n

        # 便捷：记录模板维度的“最近一次批次”与使用计数（不改变统计口径）
        if tmpl:
            self.latest_by_template[operator][tmpl] = batch
            self.template_use_counts[operator][tmpl] = (
                self.template_use_counts[operator].get(tmpl, 0) + 1
            )
            if tmpl not in self.templates_seen[operator]:
                self.templates_seen[operator].append(tmpl)

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

    # ---------------- 工具方法 ----------------

    def _ensure_op(self, operator: str) -> None:
        """确保所有容器里都有该算子的键"""
        for d in [
            self.latest,
            self.history_scores,
            self.history_counts,
            self.history_valid_counts,
        ]:
            if operator not in d:
                if d is self.latest:
                    d[operator] = None
                elif d in (self.history_counts, self.history_valid_counts):
                    d[operator] = 0
                else:
                    d[operator] = []

        if operator not in self.current_template:
            self.current_template[operator] = None
        if operator not in self.templates_seen:
            self.templates_seen[operator] = []
        if operator not in self.latest_by_template:
            self.latest_by_template[operator] = {}
        if operator not in self.template_use_counts:
            self.template_use_counts[operator] = {}
        if operator not in self.template_change_log:
            self.template_change_log[operator] = []
