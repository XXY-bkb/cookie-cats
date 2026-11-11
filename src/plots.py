"""绘图工具：所有图都带误差线/样本量，便于直接放进报告。

设计原则（也是面试里会被问的点）：
1. 图是给人看结论的，不是给统计软件看的 —— 标题直接写结论，坐标轴写清单位。
2. 比例类指标一定要画置信区间，否则读者无法判断"差 0.8pp"是真差异还是噪声。
3. 长尾分布用 ECDF + 对数横轴，而不是直方图 —— 直方图会把 99% 的样本挤成一根柱子。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端：服务器/脚本里也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# 中文字体：Windows 自带微软雅黑；macOS/Linux 缺字体时退回默认（图里会变方块，README 有说明）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号显示成方块的老问题
plt.rcParams["figure.dpi"] = 120

# 统一配色：对照组灰蓝、实验组橙红，全篇一致，读者不用反复看图例
C_CONTROL = "#4C72B0"
C_TREATMENT = "#DD8452"
C_NEUTRAL = "#8C8C8C"


def _save(fig: plt.Figure, path: str | Path, tight: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_group_sizes(counts: dict[str, int], path: str | Path) -> Path:
    """样本量对比 + SRM 参考线。样本量本身也是需要检查的指标。"""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    groups = list(counts)
    values = [counts[g] for g in groups]
    colors = [C_CONTROL if g == "gate_30" else C_TREATMENT for g in groups]
    bars = ax.bar(groups, values, color=colors, width=0.55)
    expected = sum(values) / len(values)
    ax.axhline(expected, ls="--", lw=1, color=C_NEUTRAL, label=f"均分参考线 {expected:,.0f}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(min(values) * 0.98, max(values) * 1.01)
    ax.set_ylabel("用户数")
    ax.set_title("两组样本量：几乎均分（差异 < 1%）")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_ecdf_rounds(
    control: pd.Series, treatment: pd.Series, path: str | Path, x_max: float | None = None
) -> Path:
    """两组游戏局数的 ECDF（经验累积分布），横轴对数。

    为什么用 ECDF：它把"分位数"直接画出来了 —— 曲线在 y=0.5 处的横坐标就是中位数。
    为什么 log 横轴：局数从 0 到 5 万，线性轴会把 99% 的人压在最左边一根线里。
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for series, label, color in (
        (control, "gate_30（对照）", C_CONTROL),
        (treatment, "gate_40（实验）", C_TREATMENT),
    ):
        x = np.sort(series.to_numpy())
        # 只画到 x_max，避免极少数超长尾把图拉扁；但中位数/90 分位都在范围内，不影响结论
        if x_max is not None:
            x = x[x <= x_max]
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=f"{label}，中位数 {np.median(series):.0f}", color=color, lw=1.8)
    ax.set_xscale("symlog", linthresh=1)  # 有 0 局玩家，symlog 才能同时显示 0 和 5 万
    ax.set_xlabel("累计游戏局数（对数轴）")
    ax.set_ylabel("累积用户占比")
    ax.set_title("两组游戏局数分布几乎重合：门槛改动没有改变玩家行为")
    ax.axhline(0.5, ls=":", lw=1, color=C_NEUTRAL)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_retention_bars(
    stats: dict[str, dict[str, float]], path: str | Path, title: str, ymin: float | None = None
) -> Path:
    """分组留存率柱状图 + 95% CI 误差线。

    stats 结构：{"gate_30": {"rate":..., "lo":..., "hi":...}, "gate_40": {...}}
    ymin 用来截断 y 轴（从 17.5% 起），否则两根柱子看起来"一样高"，差异被视觉抹平。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = list(stats)
    rates = [stats[g]["rate"] for g in groups]
    errs = [
        [stats[g]["rate"] - stats[g]["lo"] for g in groups],
        [stats[g]["hi"] - stats[g]["rate"] for g in groups],
    ]
    colors = [C_CONTROL if g == "gate_30" else C_TREATMENT for g in groups]
    bars = ax.bar(groups, rates, color=colors, width=0.5, yerr=errs, capsize=6,
                  error_kw={"lw": 1.2, "ecolor": "#333333"})
    for bar, g in zip(bars, groups):
        ax.text(bar.get_x() + bar.get_width() / 2, stats[g]["hi"] + 0.004,
                f"{stats[g]['rate']:.2%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(ymin if ymin is not None else 0, max(stats[g]["hi"] for g in groups) + 0.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_ylabel("7 日留存率" if "7" in title else "留存率")
    ax.set_title(title)
    if ymin is not None:
        # 放在坐标区左上角并加白底，避免压在柱子上看不清
        ax.text(0.015, 0.97, f"注意：y 轴自 {ymin:.1%} 起，柱高被放大",
                transform=ax.transAxes, fontsize=7, color="#666666",
                va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def plot_stratified(
    table: pd.DataFrame, path: str | Path, title: str = "分桶后的 7 日留存：各桶方向一致"
) -> Path:
    """分层留存折线图：横轴是局数分桶（有序），纵轴留存率，两条线 = 两组。

    table 需要列：bin（有序分类）、control_rate、treatment_rate、control_lo/hi、treatment_lo/hi
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(table))
    for prefix, label, color in (
        ("control", "gate_30（对照）", C_CONTROL),
        ("treatment", "gate_40（实验）", C_TREATMENT),
    ):
        rate = table[f"{prefix}_rate"].to_numpy()
        lo = table[f"{prefix}_lo"].to_numpy()
        hi = table[f"{prefix}_hi"].to_numpy()
        ax.plot(x, rate, marker="o", label=label, color=color, lw=1.8)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)  # 95% CI 带
    ax.set_xticks(x)
    ax.set_xticklabels(table["bin"].astype(str))
    ax.set_xlabel("累计游戏局数分桶")
    ax.set_ylabel("7 日留存率")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_title(title)
    # 每个桶的差值直接标在顶部：读者一眼看到"差多少"，不用拿尺子量两条线的间距
    if "diff_pp" in table:
        for xi, d in zip(x, table["diff_pp"]):
            ax.annotate(f"{d:+.2f}pp", (xi, 0.96), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=7, color="#555555")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_power_curve(
    p_base: float,
    n_per_group: int,
    deltas: np.ndarray,
    current_power: float,
    current_delta: float,
    path: str | Path,
) -> Path:
    """功效曲线：横轴是真实提升幅度，纵轴是当前样本量下能检出的概率。"""
    from .stats_utils import achieved_power

    fig, ax = plt.subplots(figsize=(6.5, 4))
    powers = [achieved_power(p_base, d, n_per_group) for d in deltas]
    ax.plot(deltas * 100, powers, color=C_TREATMENT, lw=1.8)
    ax.axhline(0.8, ls="--", lw=1, color=C_NEUTRAL, label="常用门槛 80%")
    ax.axvline(current_delta * 100, ls=":", lw=1, color=C_CONTROL,
               label=f"本次观测效应 {current_delta * 100:.2f}pp（功效 {current_power:.0%}）")
    ax.set_xlabel("真实绝对提升（百分点）")
    ax.set_ylabel("检验功效")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"每组 {n_per_group:,} 人时，能检出多大的提升？")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_bootstrap(observed: float, boot: np.ndarray, ci: tuple[float, float], path: str | Path) -> Path:
    """Bootstrap 抽样分布：中位数差值的"可能取值"长什么样。"""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(boot, bins=50, color=C_CONTROL, alpha=0.75)
    ax.axvline(observed, color="#222222", lw=1.5, label=f"观测差值 {observed:+.1f} 局")
    ax.axvline(ci[0], color=C_TREATMENT, ls="--", lw=1.2, label=f"95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]")
    ax.axvline(ci[1], color=C_TREATMENT, ls="--", lw=1.2)
    ax.axvline(0, color=C_NEUTRAL, ls=":", lw=1)
    ax.set_xlabel("中位数差值（对照 - 实验，局）")
    ax.set_ylabel("Bootstrap 次数")
    ax.set_title("局数中位数差值：区间跨 0，差异可以忽略")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)
