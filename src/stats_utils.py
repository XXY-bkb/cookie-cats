"""可复用的 A/B 实验统计工具。

设计原则（也是面试时能讲的东西）：

1. **一个函数只做一件事**：SRM 校验、两比例检验、bootstrap、功效分析互不耦合，
   这样任何一个都能单独写测试（见 tests/test_stats_utils.py）。
2. **返回值用 dataclass**：字段有名有义，避免"第三个返回值是 p 值"这种口头约定。
3. **随机过程必须可复现**：bootstrap 一定接收 ``seed``，同 seed 同结果。
4. **方向要写死**：所有差值统一为「对照组 - 实验组」，正数表示对照组更高。
   方向一旦不固定，结论的正负号迟早写反。
5. **不依赖 DataFrame 结构**：输入是 numpy 数组/标量，方便在 notebook、脚本、
   别的项目里复用；数据清洗归数据层，统计计算归这里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

# ---------------------------------------------------------------------------
# 1. 分流健康度：SRM（Sample Ratio Mismatch）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SRMResult:
    """卡方分流校验的结果。p 值小说明"实际分流比例"和"设计比例"差得不像随机波动。"""

    counts: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    p_value: float
    alpha: float

    @property
    def passed(self) -> bool:
        """p >= alpha 视为分流正常（没有证据说明分流坏了）。"""
        return self.p_value >= self.alpha

    @property
    def max_relative_gap(self) -> float:
        """最大的「实际 - 期望」相对偏差，用来判断统计显著是否也有业务意义。"""
        gaps = [abs(c - e) / e for c, e in zip(self.counts, self.expected) if e > 0]
        return max(gaps)

    def summary(self) -> str:
        status = "通过" if self.passed else "未通过（需排查）"
        return (
            f"SRM: chi2={self.chi2:.4f}, p={self.p_value:.4f} -> {status}; "
            f"最大相对偏差 {self.max_relative_gap:.2%}"
        )


def srm_check(
    counts: Sequence[int],
    expected_ratio: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> SRMResult:
    """对分流人数做卡方拟合优度检验。

    Parameters
    ----------
    counts
        每个组实际人数，顺序要与 ``expected_ratio`` 一致。
    expected_ratio
        设计的分流比例，例如 ``[0.5, 0.5]``；默认各组等分。
    alpha
        显著性水平。

    Notes
    -----
    样本量很大时，极小的比例偏差也会"显著"（9 万人里差 800 人就 p<0.01）。
    所以除了 p 值，一定要看 ``max_relative_gap``：本例只差 0.87%，
    属于"统计上不完美、业务上无所谓"的典型情况。
    """
    obs = np.asarray(counts, dtype=float)
    if obs.ndim != 1 or obs.size < 2:
        raise ValueError("counts 需要至少两个组的一维人数序列")
    if np.any(obs < 0):
        raise ValueError("counts 不能为负数")

    if expected_ratio is None:
        ratio = np.full(obs.size, 1.0 / obs.size)
    else:
        ratio = np.asarray(expected_ratio, dtype=float)
        if ratio.size != obs.size:
            raise ValueError("expected_ratio 长度必须与 counts 一致")
        if np.any(ratio <= 0):
            raise ValueError("expected_ratio 必须为正数")
        ratio = ratio / ratio.sum()

    expected = obs.sum() * ratio
    chi2, p_value = stats.chisquare(obs, f_exp=expected)
    return SRMResult(
        counts=tuple(int(c) for c in obs),
        expected=tuple(float(e) for e in expected),
        chi2=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# 2. 两比例检验（主指标）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProportionTest:
    """两比例检验结果。diff = 对照组比例 - 实验组比例。"""

    x_control: int
    n_control: int
    x_treatment: int
    n_treatment: int
    p_control: float
    p_treatment: float
    diff: float
    rel_lift: float
    se: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def ci_excludes_zero(self) -> bool:
        return (self.ci_low > 0) or (self.ci_high < 0)

    def summary(self) -> str:
        return (
            f"对照 {self.p_control:.2%} (n={self.n_control}) vs "
            f"实验 {self.p_treatment:.2%} (n={self.n_treatment}); "
            f"绝对差 {self.diff * 100:+.3f} pp, "
            f"95% CI [{self.ci_low * 100:+.3f}, {self.ci_high * 100:+.3f}] pp, "
            f"相对 {self.rel_lift:+.2%}, p={self.p_value:.4f}"
        )


def two_proportion_test(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> ProportionTest:
    """两独立样本比例检验（z 检验）+ Wald 置信区间。

    为什么是 z 而不是卡方：两个比例做卡方检验时，``chi2 == z**2``，
    两者等价；z 检验还能直接给出有方向的差值，便于报告"谁更好"。

    为什么同时给 CI：p 值只回答"有没有证据"，CI 才回答"差多少、可能差到哪里"。
    业务决策需要后者。
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("样本量必须为正")
    if not (0 <= x_control <= n_control and 0 <= x_treatment <= n_treatment):
        raise ValueError("成功数必须落在 [0, n] 区间")

    p_c = x_control / n_control
    p_t = x_treatment / n_treatment
    diff = p_c - p_t

    # 差值标准误：两组方差相加。这里不做"合并方差"，因为 CI 要的是差值本身的抽样分布。
    se = float(np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment))
    if se == 0:
        raise ValueError("两组比例均为 0 或 1，无法估计标准误")

    z_crit = float(stats.norm.ppf(1 - alpha / 2)) if alternative == "two-sided" else float(
        stats.norm.ppf(1 - alpha)
    )
    ci_low, ci_high = diff - z_crit * se, diff + z_crit * se

    # 检验统计量用 statsmodels：零假设下用合并比例估计 SE（pooled），与 CI 的 SE 不同，
    # 这是教科书的标准做法，也避免自己写公式写错方向。
    z_stat, p_value = proportions_ztest(
        count=[x_control, x_treatment],
        nobs=[n_control, n_treatment],
        alternative=alternative,
    )

    rel_lift = float(diff / p_t) if p_t > 0 else float("nan")
    return ProportionTest(
        x_control=x_control,
        n_control=n_control,
        x_treatment=x_treatment,
        n_treatment=n_treatment,
        p_control=p_c,
        p_treatment=p_t,
        diff=float(diff),
        rel_lift=rel_lift,
        se=se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        z=float(z_stat),
        p_value=float(p_value),
        alpha=alpha,
    )


def wilson_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """单个比例的 Wilson 区间：小样本/极端比例下比 Wald 稳（不会超出 [0,1]）。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    p = x / n
    z = float(stats.norm.ppf(1 - alpha / 2))
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def diff_ci_newcombe(
    x_control: int, n_control: int, x_treatment: int, n_treatment: int, alpha: float = 0.05
) -> tuple[float, float]:
    """两比例差值的 Newcombe hybrid score 区间（用两个 Wilson 区间拼出来）。

    作用：当比例接近 0/1 时 Wald 区间会失真，Newcombe 区间是更稳的备选。
    报告里可以当作稳健性检验："Wald 与 Newcombe 结论一致，说明不是区间方法的问题"。
    """
    l1, u1 = wilson_interval(x_control, n_control, alpha)
    l2, u2 = wilson_interval(x_treatment, n_treatment, alpha)
    p1, p2 = x_control / n_control, x_treatment / n_treatment
    diff = p1 - p2
    lower = diff - float(np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2))
    upper = diff + float(np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
    return lower, upper


# ---------------------------------------------------------------------------
# 3. Bootstrap：中位数/分位数这类"没有解析公式"的统计量
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    observed: float
    ci_low: float
    ci_high: float
    n_boot: int
    stat_name: str
    alpha: float
    # 重抽样得到的差值分布，留给画图用；不参与相等比较（np 数组没法直接比）
    boot_dist: np.ndarray | None = None

    @property
    def ci_excludes_zero(self) -> bool:
        return (self.ci_low > 0) or (self.ci_high < 0)

    def summary(self) -> str:
        return (
            f"{self.stat_name} 差值 (对照-实验) = {self.observed:+.3f}, "
            f"{1 - self.alpha:.0%} CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
            f"(B={self.n_boot})"
        )


def bootstrap_diff(
    control: Sequence[float],
    treatment: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """用 bootstrap 重抽样估计「统计量差值」的置信区间（percentile 法）。

    为什么游戏局数不能用 t 检验：``sum_gamerounds`` 是长尾分布，还有一个玩家
    打了约 5 万局。正态假设不成立，均值被单个点带偏；改用中位数 + bootstrap，
    不需要任何分布假设。

    Parameters
    ----------
    control, treatment
        两组的原始观测值（本项目里是每位玩家的游戏局数）。
    stat
        想比较的统计量，默认中位数；也可以传 ``np.mean``、``lambda x: np.quantile(x, .9)``。
    n_boot
        重抽样次数，5000 足够稳定；2000 更快，CI 端点在小数点后两位可能差一点。
    """
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("两组都不能为空")
    if n_boot < 100:
        raise ValueError("n_boot 太小，置信区间会不稳定")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        # 有放回重抽样：每次抽出一个"平行世界"的样本，再算统计量
        diffs[i] = stat(rng.choice(a, a.size, replace=True)) - stat(
            rng.choice(b, b.size, replace=True)
        )

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        observed=float(stat(a) - stat(b)),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=n_boot,
        stat_name=getattr(stat, "__name__", "stat"),
        alpha=alpha,
        boot_dist=diffs,
    )


# ---------------------------------------------------------------------------
# 4. 功效分析：这实验本来需要多大样本
# ---------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> float:
    """比例差异的效应量 Cohen's h，两比例功效分析的通用尺度。"""
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("比例必须在 [0, 1] 内")
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def required_sample_size(
    p_base: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> float:
    """给定基线比例和想检测的绝对提升，返回**每组**需要的样本量。

    例：基线 19%，想检出 +1pp，alpha=0.05、power=0.8，每组约需 2.4 万人。
    """
    if delta <= 0:
        raise ValueError("delta 必须为正（只算提升方向）")
    if p_base + delta >= 1:
        raise ValueError("p_base + delta 必须小于 1")
    effect = abs(cohens_h(p_base + delta, p_base))
    n = NormalIndPower().solve_power(
        effect_size=effect, alpha=alpha, power=power, ratio=ratio, alternative="two-sided"
    )
    return float(n)


def achieved_power(
    p_base: float,
    delta: float,
    n_per_group: float,
    alpha: float = 0.05,
) -> float:
    """在给定样本量下，检测 ``delta`` 这个差异的把握（power）有多大。"""
    effect = abs(cohens_h(p_base + delta, p_base))
    return float(
        NormalIndPower().power(
            effect_size=effect, nobs1=n_per_group, alpha=alpha, ratio=1.0, alternative="two-sided"
        )
    )


def min_detectable_delta(
    p_base: float,
    n_per_group: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """反推：现有样本量在 ``power`` 把握下，最小能可靠检出多大的绝对差异（MDE）。

    这是回答"为什么不显著"的正确姿势：不是"没效果"，而是"效果小于 MDE 时
    这个实验本来就看不清"。
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group 必须为正")

    def gap(d: float) -> float:
        return achieved_power(p_base, d, n_per_group, alpha) - power

    # power 随 delta 单调递增，直接二分求根
    return float(brentq(gap, 1e-6, 0.9 - p_base, xtol=1e-10))


# ---------------------------------------------------------------------------
# 5. 多重比较
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BonferroniResult:
    p_values: tuple[float, ...]
    alpha_adjusted: float
    reject: tuple[bool, ...]

    def summary(self) -> str:
        lines = [
            f"Bonferroni: alpha={self.alpha_adjusted:.4f}（原 0.05 / {len(self.p_values)} 个检验）"
        ]
        for i, (p, r) in enumerate(zip(self.p_values, self.reject), start=1):
            lines.append(f"  指标{i}: p={p:.4f} -> {'显著' if r else '不显著'}")
        return "\n".join(lines)


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> BonferroniResult:
    """Bonferroni 校正：检验多个指标时把显著性门槛除以指标数。

    不校正就"挑显著的讲"，就是 p-hacking——面试官一问"你测了几个指标"就露馅。
    """
    ps = tuple(float(p) for p in p_values)
    if len(ps) == 0:
        raise ValueError("p_values 不能为空")
    adj = alpha / len(ps)
    return BonferroniResult(
        p_values=ps, alpha_adjusted=adj, reject=tuple(p < adj for p in ps)
    )
