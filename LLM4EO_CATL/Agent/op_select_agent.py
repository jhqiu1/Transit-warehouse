import math
from typing import Dict, Any, Optional, Tuple, List
import random


# =======================
# 决策模块接口
# =======================
class DecisionModule:
    """决定本轮：搜哪个算子、采样几次"""

    def decide(
        self, cycle_idx: int, features: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, int]:
        raise


# policy = RandomEqualAllocator(
#     operators=operators_to_optimize,
#     max_cycles=max_cycles,
#     total_budget_samples=total_budget_samples,
#     seed=20250906,
# )


class RandomEqualAllocator(DecisionModule):
    """
    简单策略：
      - 固定 max_cycles 与 total_budget_samples（不包含 warmup）
      - 每轮随机选择一个算子
      - 采样次数 = 平均预算（余数前置）
    """

    def __init__(
        self,
        operators,
        max_cycles: int,
        total_budget_samples: int,
        seed: Optional[int] = None,
    ):
        assert max_cycles > 0 and total_budget_samples > 0
        self.ops = list(operators)
        self.max_cycles = int(max_cycles)
        self.total_budget = int(total_budget_samples)
        self.per_cycle = self.total_budget // self.max_cycles
        self.remainder = self.total_budget % self.max_cycles
        self._rng = random.Random(seed)

    def decide(
        self, cycle_idx: int, features: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, int]:
        n_samples = self.per_cycle + (1 if cycle_idx < self.remainder else 0)
        n_samples = max(1, n_samples)
        op = self._rng.choice(self.ops)
        return op, n_samples


# =========================
# UCBDecision agent
# =========================

# policy = UCBDecision(
#     operators=operators_to_optimize,
#     c=1.0,
#     seed=20250906,
# )


class UCBDecision(DecisionModule):
    """
    基于 UCB1 的算子选择：
      - 若存在未探索算子（n_i == 0），优先在这些算子中随机选择一个（纯探索）
      - 否则选择使   mean_i + c * sqrt( (2*ln T) / n_i )  最大的算子
        其中 T = sum_i n_i（这里用“历史个体数”近似总拉动次数）
      - 本实现固定 n_samples = 1
    """

    def __init__(self, operators, c: float = 1.0, seed: Optional[int] = None):
        self.ops = list(operators)
        self.c = float(c)
        self._rng = random.Random(seed)

    def decide(
        self, cycle_idx: int, features: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, int]:
        # 收集每个算子的历史统计
        n_map, mean_map = {}, {}
        cold = []
        total_pulls = 0

        for op in self.ops:
            hist = (features.get(op) or {}).get("history") or {}
            n_i = int(hist.get("n_individuals") or 0)
            mu_i = hist.get("mean")
            n_map[op] = n_i
            mean_map[op] = (
                mu_i
                if (
                    mu_i is not None
                    and not (
                        isinstance(mu_i, float)
                        and (math.isnan(mu_i) or math.isinf(mu_i))
                    )
                )
                else 0.0
            )
            total_pulls += max(0, n_i)
            if n_i == 0:
                cold.append(op)

        # 若存在未探索算子，先从中随机选一个（UCB 的探索阶段）
        if cold:
            return self._rng.choice(cold), 1

        # 避免 ln(0)
        T = max(total_pulls, 1)

        # 计算每个算子的 UCB 分数
        best_op, best_ucb = None, -float("inf")
        for op in self.ops:
            n_i = max(1, n_map[op])  # 防止除 0
            mu_i = mean_map[op]
            bonus = self.c * math.sqrt((2.0 * math.log(T)) / n_i)
            ucb = mu_i + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_op = op

        # 兜底
        if best_op is None:
            best_op = self._rng.choice(self.ops)

        return best_op, 1


class Window_UCBDecision(DecisionModule):
    """
    滑动窗口 UCB（以“当前 template”为时间窗）：
      - 若存在未探索（窗口内 n_i == 0）的算子，先随机探索一个
      - 否则选择使   mu_i + c * sqrt( (2*ln T) / n_i )  最大的算子
        其中 n_i、mu_i、T 都在“当前模板窗口”内计算
      - 若拿不到模板级特征，则回退到原来的“全历史 UCB”
    """

    def __init__(self, operators, c: float = 1.0, seed: Optional[int] = None):
        self.ops = list(operators)
        self.c = float(c)
        self._rng = random.Random(seed)

    def _num(self, x, default=0.0) -> float:
        try:
            v = float(x)
            return v if math.isfinite(v) else float(default)
        except Exception:
            return float(default)

    def _window_stats_from_templates(
        self, op_feat: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        从 features[op] 中提取“当前模板窗口”的 n_i 与 mu_i。
        约定：
          - n_i = 当前模板的 use_count（窗口内“批次数”）
          - mu_i = 当前模板最近一批的 mean_score（窗口的最新均值代表）
        若字段不存在则返回 (-1, 0.0) 表示“不支持窗口”，外层将回退到全历史。
        """
        cur = (op_feat or {}).get("current_template")
        by_t = (op_feat or {}).get("by_template_meta") or {}
        if not cur or cur not in by_t:
            return -1, 0.0  # 无法做窗口

        cur_info = by_t[cur] or {}
        latest = cur_info.get("latest_batch") or {}
        n_i = int(cur_info.get("use_count") or 0)  # 窗口内“拉动次数”（批次数）
        mu_i = self._num(latest.get("mean_score"), 0.0)

        # 如果窗口内还没任何批次，则 n_i==0，表示“未探索窗口”
        return n_i, mu_i

    def _global_stats_fallback(self, op_feat: Dict[str, Any]) -> Tuple[int, float]:
        """
        回退：使用算子“全历史”统计（你的原实现）
          - n_i = history.n_individuals
          - mu_i = history.mean
        """
        hist = (op_feat or {}).get("history") or {}
        n_i = int(hist.get("n_individuals") or 0)
        mu_i_raw = hist.get("mean")
        mu_i = (
            mu_i_raw
            if (
                mu_i_raw is not None
                and isinstance(mu_i_raw, (int, float))
                and math.isfinite(float(mu_i_raw))
            )
            else 0.0
        )
        return n_i, mu_i

    def decide(
        self, cycle_idx: int, features: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, int]:
        # 收集每个算子的“模板窗口”统计；若拿不到则回退到全历史
        n_map, mean_map = {}, {}
        cold, total_pulls = [], 0

        for op in self.ops:
            op_feat = features.get(op) or {}

            # 优先用“当前模板窗口”的 n_i / mu_i
            n_i, mu_i = self._window_stats_from_templates(op_feat)

            # 如果不支持窗口（返回 n_i==-1），回退到全历史
            if n_i == -1:
                n_i, mu_i = self._global_stats_fallback(op_feat)

            n_map[op] = int(max(0, n_i))
            mean_map[op] = float(mu_i)
            total_pulls += n_map[op]
            if n_map[op] == 0:
                cold.append(op)

        # 若存在“未探索窗口”的算子，先随机探索（UCB 的探索阶段）
        if cold:
            return self._rng.choice(cold), 1

        # 避免 ln(0)
        T = max(total_pulls, 1)

        # 计算每个算子的 UCB 分数（在“模板窗口”口径下）
        best_op, best_ucb = None, -float("inf")
        for op in self.ops:
            n_i = max(1, n_map[op])  # 防止除 0
            mu_i = mean_map[op]
            bonus = self.c * math.sqrt((2.0 * math.log(T)) / n_i)
            ucb = mu_i + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_op = op

        if best_op is None:
            best_op = self._rng.choice(self.ops)

        return best_op, 1


class Win_UCBDecision_discount(DecisionModule):
    """
    滑动窗口 UCB（以“当前 template”为时间窗）：
      - 若存在未探索（窗口内 n_i == 0）的算子，先随机探索一个
      - 否则选择使   mu_i + c * sqrt( (2*ln T) / n_i )  最大的算子
        其中 n_i、mu_i、T 都在“当前模板窗口”内计算
      - 若拿不到模板级特征，则回退到原来的“全历史 UCB”
    """

    def __init__(self, operators, c: float = 1.0, seed: Optional[int] = None):
        self.ops = list(operators)
        self.c = float(c)
        self._rng = random.Random(seed)

    def _num(self, x, default=0.0) -> float:
        try:
            v = float(x)
            return v if math.isfinite(v) else float(default)
        except Exception:
            return float(default)

    def _window_stats_from_templates(
        self, op_feat: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        从 features[op] 中提取“当前模板窗口”的 n_i 与 mu_i。
        约定：
          - n_i = 当前模板的 use_count（窗口内“批次数”）
          - mu_i = 当前模板最近一批的 mean_score（窗口的最新均值代表）
        若字段不存在则返回 (-1, 0.0) 表示“不支持窗口”，外层将回退到全历史。
        """
        cur = (op_feat or {}).get("current_template")
        by_t = (op_feat or {}).get("by_template_meta") or {}
        if not cur or cur not in by_t:
            return -1, 0.0  # 无法做窗口

        cur_info = by_t[cur] or {}
        latest = cur_info.get("latest_batch") or {}
        n_i = int(cur_info.get("use_count") or 0)  # 窗口内“拉动次数”（批次数）
        mu_i = self._num(latest.get("mean_score"), 0.0)

        # 如果窗口内还没任何批次，则 n_i==0，表示“未探索窗口”
        return n_i, mu_i

    def _global_stats_fallback(self, op_feat: Dict[str, Any]) -> Tuple[int, float]:
        """
        回退：使用算子“全历史”统计（你的原实现）
          - n_i = history.n_individuals
          - mu_i = history.mean
        """
        hist = (op_feat or {}).get("history") or {}
        n_i = int(hist.get("n_individuals") or 0)
        mu_i_raw = hist.get("mean")
        mu_i = (
            mu_i_raw
            if (
                mu_i_raw is not None
                and isinstance(mu_i_raw, (int, float))
                and math.isfinite(float(mu_i_raw))
            )
            else 0.0
        )
        return n_i, mu_i

    def _compute_weighted_stats(self, op, stor, decay=0.4):
        """
        结合 storage 中不同模板的历史窗口数据，计算带权均值与带权样本数。
        """
        tmpl_list = stor.templates_seen.get(op, [])
        if not tmpl_list:
            return None, None  # 没有历史记录

        weights, mus, counts = [], [], []
        K = len(tmpl_list)
        for i, tmpl in enumerate(tmpl_list):
            # 时间越近权重越大
            w = math.exp(-decay * (K - 1 - i))
            mu = getattr(stor.latest_by_template[op][tmpl], "mean_score", 0.0)
            n = stor.template_use_counts[op][tmpl]
            weights.append(w)
            mus.append(mu)
            counts.append(n)

        total_w = sum(weights)
        weighted_mu = sum(w * m for w, m in zip(weights, mus)) / total_w
        weighted_n = sum(w * n for w, n in zip(weights, counts))
        return weighted_mu, max(1.0, weighted_n)

    def decide(
        self, cycle_idx: int, features: Dict[str, Dict[str, Any]], stor
    ) -> Tuple[str, int]:
        # 收集每个算子的“模板窗口”统计；若拿不到则回退到全历史
        n_map, mean_map = {}, {}
        cold, total_pulls = [], 0

        for op in self.ops:
            op_feat = features.get(op) or {}

            # 优先用“当前模板窗口”的 n_i / mu_i
            mu_i, n_i = self._compute_weighted_stats(op, stor, decay=0.4)

            # 如果不支持窗口（返回 n_i==-1），回退到全历史
            if n_i == -1:
                n_i, mu_i = self._global_stats_fallback(op_feat)

            n_map[op] = int(max(0, n_i))
            mean_map[op] = float(mu_i)
            total_pulls += n_map[op]
            if n_map[op] == 0:
                cold.append(op)

        # 若存在“未探索窗口”的算子，先随机探索（UCB 的探索阶段）
        if cold:
            return self._rng.choice(cold), 1

        # 避免 ln(0)
        T = max(total_pulls, 1)

        # 计算每个算子的 UCB 分数（在“模板窗口”口径下）
        best_op, best_ucb = None, -float("inf")
        for op in self.ops:
            n_i = max(1, n_map[op])  # 防止除 0
            mu_i = mean_map[op]
            bonus = self.c * math.sqrt((2.0 * math.log(T)) / n_i)
            ucb = mu_i + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_op = op

        if best_op is None:
            best_op = self._rng.choice(self.ops)

        return best_op, 1


# =========================
# LLM agent with history
# =========================

import json, math, re
from typing import Dict, Any, Tuple, List, Optional


def _safe_float(x, default=None):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _pct(a, nd=2):
    v = _safe_float(a, None)
    return f"{v*100:.{nd}f}%" if v is not None else "null"


def _fmt(a, nd=6):
    v = _safe_float(a, None)
    return f"{v:.{nd}f}" if v is not None else "null"


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\{.*\})", text, re.S)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


# ---------- Prompt 生成 ----------
def build_decision_prompt(
    cycle_idx: int,
    operators: List[str],
    features: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Any]],
    objective_desc: str = "Maximize multi-objective hypervolume (HV) while balancing convergence and solution diversity",
    rules_hint: Optional[str] = None,
    max_hist: int = 5,
) -> str:
    """
    Build a detailed prompt from storages.build_features() + decision history.
    （最小修改：将 n_samples 替换为 update_template）
    """
    # 最近几轮历史
    hist_lines = []
    for h in history[-max_hist:]:
        hist_lines.append(
            f"Cycle {h['cycle']}: op={h['op']}, mean={_fmt(h.get('mean'))}, best={_fmt(h.get('best'))}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "No prior decisions."

    # 特征信息（保持原样）
    lines = []
    for op in operators:
        f = features.get(op, {})
        latest = f.get("latest") or {}
        hist = f.get("history") or {}

        l_mean = latest.get("mean_score")
        l_best = latest.get("best_score")
        l_worst = latest.get("worst_score")
        l_std = latest.get("std_score")
        l_q10 = latest.get("q_low")
        l_q90 = latest.get("q_high")
        l_vr = latest.get("valid_rate")

        h_n = hist.get("n_individuals")
        h_mean = hist.get("mean")
        h_best = hist.get("best")
        h_worst = hist.get("worst")
        h_std = hist.get("std")
        h_q10 = hist.get("q_low")
        h_q90 = hist.get("q_high")
        h_vr = hist.get("valid_rate")
        h_ucb = hist.get("ucb")
        h_lcb = hist.get("lcb")

        d_mean = (
            _safe_float(l_mean, None) - _safe_float(h_mean, 0.0)
            if l_mean is not None and h_mean is not None
            else None
        )
        d_best = (
            _safe_float(l_best, None) - _safe_float(h_best, 0.0)
            if l_best is not None and h_best is not None
            else None
        )

        lines.append(
            f"- operator: {op}\n"
            f"  Latest batch: mean={_fmt(l_mean)}, best={_fmt(l_best)}, worst={_fmt(l_worst)}, "
            f"std={_fmt(l_std)}, q10={_fmt(l_q10)}, q90={_fmt(l_q90)}, valid_rate={_pct(l_vr)}\n"
            f"  Historical aggregate: n={h_n}, mean={_fmt(h_mean)}, best={_fmt(h_best)}, worst={_fmt(h_worst)}, "
            f"std={_fmt(h_std)}, q10={_fmt(h_q10)}, q90={_fmt(h_q90)}, valid_rate={_pct(h_vr)}, "
            f"ucb={_fmt(h_ucb)}, lcb={_fmt(h_lcb)}\n"
            f"  Trend: Δmean(latest - history)={_fmt(d_mean)}, Δbest={_fmt(d_best)}"
        )
    features_block = "\n".join(lines)

    rules = rules_hint or (
        "1) Prefer operators with higher mean and best scores and with higher valid_rate; "
        "2) Allow exploration for operators with fewer samples if valid_rate is acceptable; "
        "3) Favor operators with positive Δmean/Δbest in the latest batch; "
        "4) Use UCB/LCB if available to avoid premature convergence; "
        "5) Avoid repeating the same operator too many times consecutively unless strongly improving."
        "6) Modifying the template will recalibrate the algorithm to generate distributions. If you believe the current template offers little value for continued search, you may return true or false."
    )

    # ---------- 这里做最小改动：把 n_samples 改为 update_template ----------
    schema = (
        "{\n"
        f'  "op": "<must be one of: {", ".join(operators)}>",\n'
        '  "update_template": true | false,\n'
        '  "reason": "<1-2 sentences explanation>"\n'
        "}"
    )

    prompt = (
        "You are a decision assistant for allocating search budget among evolutionary operators.\n"
        f"[Problem context] In an NSGA-III evolutionary framework for Flexible Job Shop Scheduling (FJSP), "
        f"we iteratively select ONE operator to improve, generate candidates via an LLM, and evaluate them by hypervolume (HV). "
        f"The objective is: {objective_desc}.\n\n"
        f"[Current iteration] cycle = {cycle_idx}\n\n"
        "[Recent decision history]\n"
        f"{history_block}\n\n"
        "[Candidate operators and statistics]\n"
        f"{features_block}\n\n"
        "[Decision guidelines]\n"
        f"{rules}\n\n"
        "[Output requirements]\n"
        "Return ONLY a SINGLE JSON object:\n"
        f"{schema}\n"
    )
    return prompt


# ---------- 校验 ----------
def validate_llm_decision(d: dict, allowed_ops: List[str]) -> Tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "Output is not a JSON object."
    if "op" not in d:
        return False, "Missing field 'op'."
    if d["op"] not in allowed_ops:
        return False, f"'op' is not in allowed set {allowed_ops}."
    # ---------- 这里替换 ----------
    if "update_template" not in d:
        return False, "Missing field 'update_template'."
    if not isinstance(d["update_template"], bool):
        return False, "'update_template' must be boolean (true/false)."
    # 原实现未强制要求 reason，但如果你需要可以在这里加上检查
    return True, ""


# ---------- 策略类 ----------
class LLMDecisionPolicy:
    def __init__(self, llm, operators: List[str], retry: int = 1):
        self.llm = llm
        self.operators = list(operators)
        self.max_retry = max(0, int(retry))
        self.history: List[Dict[str, Any]] = []

    def decide(
        self, cycle_idx: int, features: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, int]:
        prompt = build_decision_prompt(
            cycle_idx=cycle_idx,
            operators=self.operators,
            features=features,
            history=self.history,
        )
        # print(prompt)
        # return 0, 0

        attempts = 0
        while True:
            raw = self.llm.draw_sample(prompt) or ""
            parsed = _extract_json(raw) or {}
            ok, err = validate_llm_decision(parsed, self.operators)
            if ok:
                op = parsed["op"]
                n = max(1, int(parsed.get("n_samples", 1)))
                # 记录历史
                f = features.get(op, {})
                self.history.append(
                    {
                        "cycle": cycle_idx,
                        "op": op,
                        "mean": f.get("latest", {}).get("mean_score"),
                        "best": f.get("latest", {}).get("best_score"),
                    }
                )
                return op, n
            attempts += 1
            if attempts > self.max_retry:
                # fallback: choose operator with fewest samples but valid_rate not low
                min_op, min_count = None, float("inf")
                for op, f in features.items():
                    count = f.get("history", {}).get("n_individuals") or 0
                    vr = f.get("history", {}).get("valid_rate") or 0
                    if count < min_count and vr > 0.3:  # threshold可调
                        min_op, min_count = op, count
                if not min_op:  # 全部无效就随机
                    import random

                    min_op = random.choice(self.operators)
                self.history.append({"cycle": cycle_idx, "op": min_op})
                return min_op, 1
            # 否则 retry: 在 prompt 后加 hint
            prompt = (
                prompt
                + f"\n[Error correction] Previous output invalid: {err}. Please strictly follow JSON schema."
            )


# decision_rewrite_ucb_min.py
import math, random
from typing import Any, Dict, Optional, Tuple


def _num(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return default


class TemplateRewriteDecision:
    """
    是否让 LLM 改写模板（REWRITE）还是保持（KEEP）
    只依赖：
      - stor.get_current_template_features(operator)
      - stor.get_all_templates_features(operator)
    UCB1: score = r + c * sqrt((2*ln T)/n)
    """

    def __init__(self, c: float = 1.0, seed: Optional[int] = None):
        self.c = float(c)
        self._rng = random.Random(seed)

    def decide(self, cycle_idx: int, operator: str, stor) -> Tuple[bool, int]:
        """
        返回: (do_rewrite, 1)
          do_rewrite=True  -> 让 LLM 改写模板
          do_rewrite=False -> 保持当前模板
        """
        cur = stor.get_current_template_features(operator) or {}
        allf = stor.get_all_templates_features(operator) or {}

        cur_tmpl = cur.get("current_template")
        cur_latest = (cur.get("latest_of_current_template") or {}) if cur_tmpl else {}
        cur_meta = cur.get("template_meta") or {}

        # 无当前模板或还未产生 batch：先改写产一个
        if not cur_tmpl or not cur_latest:
            return True, 1

        # KEEP 臂
        r_keep = _num(cur_latest.get("mean_score"), 0.0)
        n_keep = int(cur_meta.get("use_count") or 0)

        # REWRITE 臂（基于“其它模板”的先验）
        r_sum, r_cnt, n_rewrite = 0.0, 0, 0
        for tmpl, info in (allf or {}).items():
            if tmpl == cur_tmpl:
                continue
            latest = (info or {}).get("latest_batch") or {}
            r = _num(latest.get("mean_score"), None)
            if r is not None:
                r_sum += r
                r_cnt += 1
            n_rewrite += int((info or {}).get("use_count") or 0)

        r_rewrite = (r_sum / r_cnt) if r_cnt > 0 else 0.0

        # 从未尝试过其它模板：给 REWRITE 一次机会
        if n_rewrite == 0:
            return True, 1

        # UCB 比较
        T = max(n_keep + n_rewrite, 1)
        lnT = math.log(T)
        ucb_keep = r_keep + self.c * math.sqrt((2.0 * lnT) / max(1, n_keep))
        ucb_rewr = r_rewrite + self.c * math.sqrt((2.0 * lnT) / max(1, n_rewrite))

        do_rewrite = ucb_rewr >= ucb_keep
        return do_rewrite, 1


class TemplateRuleDecision:
    """ """

    def __init__(self, c: float = 1.0, seed: Optional[int] = None):
        self.c = float(c)
        self._rng = random.Random(seed)

    def decide(self, cycle_idx: int, operator: str, stor) -> Tuple[bool, int]:
        """
        返回: (do_rewrite, 1)
        do_rewrite=True  -> 让 LLM 改写模板
        do_rewrite=False -> 保持当前模板
        规则：
        1) 最新一批 valid_rate < 1.0 则改写；
        2) 否则若最新(best & worst)均 >= 历史均值，则 KEEP；否则 REWRITE；
        3) 缺字段时保守 KEEP。
        """
        cur = stor.get_current_template_features(operator) or {}
        allf = stor.get_all_templates_features(operator) or {}

        # 最新一批（当前模板）
        latest = cur.get("latest_of_current_template") or {}
        valid_rate = _num(latest.get("valid_rate"), None)

        # 先判断是否出现无效
        if valid_rate is not None and valid_rate < 0.9:
            return True, 1  # 产生过无效 → 改写

        # 取最新批的极值；缺失则回退到 mean_score
        best_now = _num(latest.get("best_score"), None)
        worst_now = _num(latest.get("worst_score"), None)
        mean_now = _num(latest.get("mean_score"), None)
        if best_now is None:
            best_now = mean_now
        if worst_now is None:
            worst_now = mean_now

        # 历史平均（优先从 cur 的历史字段里拿）
        # 兼容不同命名：history.mean / history_mean / hist_mean
        history = cur.get("history") or {}
        hist_mean = _num(history.get("mean"), None)
        if hist_mean is None:
            hist_mean = _num(cur.get("history_mean"), None)
        if hist_mean is None:
            hist_mean = _num(cur.get("hist_mean"), None)

        # 字段不全：保守 KEEP
        if best_now is None or worst_now is None or hist_mean is None:
            return False, 1

        # 对比规则：最新一批的最好&最差都不低于历史均值 → KEEP，否则 REWRITE
        keep = (best_now >= hist_mean) and (worst_now >= hist_mean)
        return (not keep), 1


class TemplateRuleDecision_limited:
    """基于规则 + 最小采样长度限制的模板改写决策"""

    def __init__(
        self, c: float = 1.0, seed: Optional[int] = None, min_samples: int = 25
    ):
        """
        参数:
            c: 控制参数（目前未用）
            seed: 随机种子
            min_samples: 每个模板的最小采样长度限制
        """
        self.c = float(c)
        self.min_samples = int(min_samples)
        self._rng = random.Random(seed)

    def decide(self, cycle_idx: int, operator: str, stor) -> Tuple[bool, int]:
        """
        返回: (do_rewrite, 1)
        do_rewrite=True  -> 让 LLM 改写模板
        do_rewrite=False -> 保持当前模板
        """
        cur = stor.get_current_template_features(operator) or {}
        latest = cur.get("latest_of_current_template") or {}
        meta = cur.get("template_meta") or {}

        # ==== Step 1: 计算模板累计采样数量 ====
        use_count = _num(meta.get("use_count"), 0)
        n_samples = _num(latest.get("n_samples"), 0)
        num_samples = (use_count or 0) * (n_samples or 0)

        if num_samples < self.min_samples:
            # 样本未达到最小采样长度 → 禁止改写
            return False, 1

        # ==== Step 2: 判断是否无效 ====
        valid_rate = _num(latest.get("valid_rate"), None)
        if valid_rate is not None and valid_rate < 0.9:
            return True, 1  # 出现无效 → 改写

        # ==== Step 3: 提取最新批次性能 ====
        best_now = _num(latest.get("best_score"), None)
        worst_now = _num(latest.get("worst_score"), None)
        mean_now = _num(latest.get("mean_score"), None)
        if best_now is None:
            best_now = mean_now
        if worst_now is None:
            worst_now = mean_now

        # ==== Step 4: 获取历史均值 ====
        history = stor.history(operator) or {}
        hist_mean = _num(history.get("mean"), None)
        if hist_mean is None:
            hist_mean = _num(cur.get("history_mean"), None)
        if hist_mean is None:
            hist_mean = _num(cur.get("hist_mean"), None)

        # ==== Step 5: 字段缺失 → 保守 KEEP ====
        if best_now is None or worst_now is None or hist_mean is None:
            return False, 1

        # ==== Step 6: 判断是否改写 ====
        keep = (best_now >= hist_mean) and (worst_now >= hist_mean)
        return (not keep), 1
