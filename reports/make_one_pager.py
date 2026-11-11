"""把结果渲染成一页 A4 结论图（PNG + PDF）。

用法：python -m reports.make_one_pager  （或 python reports/make_one_pager.py）

设计思路（这也是"一页结论"的通用套路）：
1. 标题就是结论，读者不看正文也知道答案；
2. 三张图各承担一个论点：效应有多大、行为有没有变、分层后结论稳不稳；
3. 文字只留"结论 / 建议 / 局限"三块，每块不超过三行。
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.plots import C_CONTROL, C_TREATMENT  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "raw" / "cookie_cats.csv"


def wrap(text: str, width: int = 82) -> str:
    """按字数硬折行。matplotlib 不会自动换行，中文还按字符宽度算，必须自己折。"""
    return "\n".join(textwrap.wrap(text, width=width))


def main() -> None:
    r = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    df = pd.read_csv(DATA)
    for c in ("retention_1", "retention_7"):
        df[c] = df[c].astype(int)

    p, s, sec = r["primary"], r["stratified"], r["secondary"]
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 竖版，单位英寸

    # ---------- 标题：一句话结论 ----------
    ax = fig.add_axes([0.09, 0.885, 0.84, 0.10])
    ax.axis("off")
    ax.text(0.0, 0.95, "30 关解锁优于 40 关：7 日留存高 0.82pp，且未损害前期体验",
            fontsize=14.5, fontweight="bold", va="top")
    ax.text(0.0, 0.42,
            f"数据：Cookie Cats 90,189 名新用户 · 14 天窗口 · 主指标 7 日留存 · 两比例 z 检验（α=0.05）\n"
            f"结论：gate_30 {p['control_rate']:.2%} vs gate_40 {p['treatment_rate']:.2%}，"
            f"绝对差 {p['diff_pp']:+.2f}pp，95% CI [{p['ci_low_pp']:+.2f}, {p['ci_high_pp']:+.2f}]pp，"
            f"p={p['p_value']:.4f}",
            fontsize=8, color="#333333", va="top", linespacing=1.55)

    # ---------- 图 1：主指标 + 95% CI ----------
    ax = fig.add_axes([0.10, 0.745, 0.80, 0.135])
    rates = [p["control_rate"], p["treatment_rate"]]
    errs = [
        [p["control_rate"] - p["wilson_control"][0], p["treatment_rate"] - p["wilson_treatment"][0]],
        [p["wilson_control"][1] - p["control_rate"], p["wilson_treatment"][1] - p["treatment_rate"]],
    ]
    bars = ax.bar(["gate_30（对照）", "gate_40（实验）"], rates, color=[C_CONTROL, C_TREATMENT],
                  width=0.36, yerr=errs, capsize=5, error_kw={"lw": 1.1, "ecolor": "#333"})
    for b, v, hi in zip(bars, rates, (p["wilson_control"][1], p["wilson_treatment"][1])):
        ax.text(b.get_x() + b.get_width() / 2, hi + 0.0012, f"{v:.2%}", ha="center", fontsize=10.5)
    ax.set_ylim(0.175, 0.199)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.tick_params(labelsize=8)
    ax.set_title("① 主指标：7 日留存显著更高（误差线 95% CI；y 轴自 17.5% 起，柱高放大）",
                 fontsize=9.5, loc="left", pad=6)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- 图 2：局数 ECDF ----------
    ax = fig.add_axes([0.10, 0.565, 0.80, 0.135])
    for ver, label, color in (("gate_30", "gate_30", C_CONTROL), ("gate_40", "gate_40", C_TREATMENT)):
        x = np.sort(df.loc[df.version == ver, "sum_gamerounds"].to_numpy())
        x = x[x <= 5000]
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, lw=1.5,
                label=f"{label}（中位数 {np.median(df.loc[df.version == ver, 'sum_gamerounds']):.0f} 局）")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_ylabel("累积用户占比", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("② 游戏局数（横轴对数）：两条 ECDF 几乎重合，改动没有改变玩家行为",
                 fontsize=9.5, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- 图 3：分层 ----------
    ax = fig.add_axes([0.10, 0.375, 0.80, 0.135])
    t = pd.DataFrame(s["table"])
    x = np.arange(len(t))
    for prefix, label, color in (
        ("control", "gate_30", C_CONTROL),
        ("treatment", "gate_40", C_TREATMENT),
    ):
        ax.plot(x, t[f"{prefix}_rate"], marker="o", ms=3.5, lw=1.5, color=color, label=label)
        ax.fill_between(x, t[f"{prefix}_lo"], t[f"{prefix}_hi"], color=color, alpha=0.15, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(t["bin"].astype(str), fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("累计游戏局数分桶", fontsize=8)
    ax.set_ylabel("7 日留存率", fontsize=8)
    ax.set_title("③ 分层：差异集中在 21-100 局的中度玩家，交互检验 p=0.09 不显著",
                 fontsize=9.5, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- 文字：结论 / 建议 / 局限 ----------
    lines = [
        "结论",
        f"1. 主指标成立：7 日留存 {p['control_rate']:.2%} vs {p['treatment_rate']:.2%}，差 "
        f"{p['diff_pp']:+.2f}pp（相对 {p['rel_lift']:.1%}），p={p['p_value']:.4f}，CI 与 Newcombe 区间都不含 0。",
        f"2. 无附带损害：1 日留存差 {sec['retention_1']['diff_pp']:+.2f}pp（p={sec['retention_1']['p_value']:.3f}，"
        f"校正后不显著）；局数中位数差 {sec['rounds_median_diff']['observed']:+.0f} 局，"
        f"95% CI [{sec['rounds_median_diff']['ci'][0]:+.0f}, {sec['rounds_median_diff']['ci'][1]:+.0f}]，可忽略。",
        f"3. 结论稳健：按对照组结构标准化后差 {s['standardized']['diff_pp']:+.2f}pp；"
        f"5 个分层中 4 个方向一致，交互检验 p={s['interaction_p']:.2f}，没有异质性证据。",
        "建议",
        "• 保持 gate_30（30 关解锁），不切换到 gate_40；每 10 万新用户预计多留住约 820 人（7 日口径）。",
        f"• 14 天窗口只看短期，若要外推到长期留存/收入，应延长观测后复测；"
        f"下次想稳定检出 0.5pp，每组约需 {r['power']['required_n']['0.5pp']:,} 人"
        f"（本次 {r['power']['n_per_group']:,}，MDE {r['power']['mde_pp']:.2f}pp）。",
        "局限",
        f"• 仅一款游戏、单一市场、14 天；只测留存，未测收入。SRM 未通过（p={r['health']['srm']['p_value']:.4f}，"
        f"两组差 0.87%），分流环节需排查，但对 0.8pp 结论影响有限。",
        "• 分桶变量“局数”受版本影响（治疗后变量），分层只作描述性分解，不能当因果解释。",
    ]
    ax = fig.add_axes([0.09, 0.015, 0.84, 0.292])
    ax.axis("off")
    y = 1.0
    for line in lines:
        if line in ("结论", "建议", "局限"):
            ax.text(0.0, y, line, fontsize=9.5, fontweight="bold", va="top")
            y -= 0.068
        else:
            wrapped = wrap(line, 84)
            ax.text(0.0, y, wrapped, fontsize=7.2, va="top", linespacing=1.5, color="#222222")
            n = max(1, wrapped.count("\n") + 1)
            y -= 0.043 * n + 0.010

    fig.text(0.09, 0.004,
             "可复现代码：src/stats_utils.py + src/analysis.py（python -m src.analysis 重跑全部数字）",
             fontsize=6.2, color="#777")

    out_png = REPORTS / "one_pager.png"
    out_pdf = REPORTS / "one_pager.pdf"
    fig.savefig(out_png, dpi=200, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    print("saved", out_png, out_pdf)


if __name__ == "__main__":
    main()
