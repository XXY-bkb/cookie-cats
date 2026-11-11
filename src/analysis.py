"""可复现的分析管道：一条命令跑完所有统计，输出 JSON + 图。

用法（在项目根目录）：
    python -m src.analysis

为什么要有这个文件，而不是只写 notebook：
- notebook 适合"讲故事"，但单元格顺序乱了结果就不可复现；
- 管道适合"算数"，同样的输入永远给同样的输出，测试也好写；
- 报告只消费 results.json 和 figures/，数据和展示彻底分开。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .plots import (
    plot_bootstrap,
    plot_ecdf_rounds,
    plot_group_sizes,
    plot_power_curve,
    plot_retention_bars,
    plot_stratified,
)
from .stats_utils import (
    achieved_power,
    bonferroni,
    bootstrap_diff,
    diff_ci_newcombe,
    min_detectable_delta,
    required_sample_size,
    srm_check,
    two_proportion_test,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cookie_cats.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

CONTROL, TREATMENT = "gate_30", "gate_40"
ALPHA = 0.05


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # retention 读进来是 bool，转成 0/1 才能做比例运算
    for col in ("retention_1", "retention_7"):
        df[col] = df[col].astype(int)
    return df


def section_health(df: pd.DataFrame) -> dict:
    hr("1. 数据体检")
    print(f"形状 {df.shape}，列 {list(df.columns)}")
    print(f"缺失值：{int(df.isna().sum().sum())}，重复 userid：{int(df.userid.duplicated().sum())}")
    print(f"局数：min={df.sum_gamerounds.min()}, max={df.sum_gamerounds.max()}, "
          f"median={df.sum_gamerounds.median():.0f}, mean={df.sum_gamerounds.mean():.1f}")
    counts = df["version"].value_counts().to_dict()
    print(f"分组样本量：{counts}")

    srm = srm_check([counts[CONTROL], counts[TREATMENT]])
    print(srm.summary())

    # 极端值画像：这些人是谁？他们的留存和普通人一样吗？
    top = df.nlargest(3, "sum_gamerounds")[["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]]
    print("\n局数最多的 3 个用户（异常值诊断用）：")
    print(top.to_string(index=False))

    plot_group_sizes(counts, FIGURES / "01_group_sizes.png")
    return {
        "shape": list(df.shape),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.userid.duplicated().sum()),
        "counts": {k: int(v) for k, v in counts.items()},
        "rounds_max": int(df.sum_gamerounds.max()),
        "rounds_median": float(df.sum_gamerounds.median()),
        "rounds_mean": float(df.sum_gamerounds.mean()),
        "srm": {"chi2": srm.chi2, "p_value": srm.p_value, "passed": srm.passed,
                "max_relative_gap": srm.max_relative_gap},
    }


def section_primary(df: pd.DataFrame) -> dict:
    hr("2. 主指标：7 日留存")
    c = df[df.version == CONTROL]
    t = df[df.version == TREATMENT]
    test = two_proportion_test(
        int(c.retention_7.sum()), len(c), int(t.retention_7.sum()), len(t), alpha=ALPHA
    )
    print(test.summary())
    print(f"显著吗（p<{ALPHA}）：{test.significant}；CI 是否排除 0：{test.ci_excludes_zero}")

    # 每个组各自的 Wilson 区间：用于画柱状图的误差线
    wc = wilson_interval(int(c.retention_7.sum()), len(c))
    wt = wilson_interval(int(t.retention_7.sum()), len(t))
    print(f"Wilson 95% CI：对照 [{wc[0]:.2%}, {wc[1]:.2%}]，实验 [{wt[0]:.2%}, {wt[1]:.2%}]")

    # 稳健性：换一种区间算法，结论会不会变
    nc = diff_ci_newcombe(int(c.retention_7.sum()), len(c), int(t.retention_7.sum()), len(t))
    print(f"Newcombe 区间（稳健性备选）：[{nc[0] * 100:+.3f}, {nc[1] * 100:+.3f}] pp")

    plot_retention_bars(
        {
            CONTROL: {"rate": test.p_control, "lo": wc[0], "hi": wc[1]},
            TREATMENT: {"rate": test.p_treatment, "lo": wt[0], "hi": wt[1]},
        },
        FIGURES / "02_retention7_ci.png",
        title="7 日留存：gate_30 显著更高（约 +0.8pp）",
        ymin=0.175,
    )
    return {
        "control_rate": test.p_control,
        "treatment_rate": test.p_treatment,
        "diff_pp": test.diff * 100,
        "rel_lift": test.rel_lift,
        "ci_low_pp": test.ci_low * 100,
        "ci_high_pp": test.ci_high * 100,
        "z": test.z,
        "p_value": test.p_value,
        "significant": bool(test.significant),
        "newcombe_ci_pp": [nc[0] * 100, nc[1] * 100],
        "wilson_control": list(wc),
        "wilson_treatment": list(wt),
    }


def section_secondary(df: pd.DataFrame) -> dict:
    hr("3. 次要指标：1 日留存 + 游戏局数")
    c = df[df.version == CONTROL]
    t = df[df.version == TREATMENT]

    d1 = two_proportion_test(int(c.retention_1.sum()), len(c), int(t.retention_1.sum()), len(t))
    print("1 日留存：", d1.summary())

    # 局数：中位数用 bootstrap，均值用 t 检验 —— 后者是常见错误，故意跑出来做对比
    med = bootstrap_diff(c.sum_gamerounds.to_numpy(), t.sum_gamerounds.to_numpy(), n_boot=2000, seed=42)
    print("局数中位数差值：", med.summary())

    ttest = stats.ttest_ind(c.sum_gamerounds, t.sum_gamerounds, equal_var=False)
    print(f"（反面教材）对局数均值做 Welch t 检验：t={ttest.statistic:.2f}, p={ttest.pvalue:.3f} "
          f"—— 均值被 5 万局的极端玩家主导，这个 p 值没有业务意义")

    q = [0.5, 0.9, 0.99, 0.999]
    print("\n分位数对比（局数）：")
    quant = pd.DataFrame({
        CONTROL: c.sum_gamerounds.quantile(q),
        TREATMENT: t.sum_gamerounds.quantile(q),
    })
    print(quant.to_string())

    # 异常值敏感性：去掉局数最大的 1 个 / 每组 0.1% 用户，中位数结论变不变
    print("\n异常值敏感性（局数中位数）：")
    sensitivity = {}

    def drop_top(s: pd.Series, k: int) -> pd.Series:
        return s.sort_values().iloc[:-k] if k > 0 else s

    for key, label, k in (
        ("raw", "原始", 0),
        ("drop_top1", "去掉最大 1 人", 1),
        ("drop_top_0.1pct", "去掉最大 0.1%", max(1, int(len(df) * 0.001))),
    ):
        cc, tt = drop_top(c.sum_gamerounds, k), drop_top(t.sum_gamerounds, k)
        sensitivity[key] = {"label": label, "control_median": float(cc.median()),
                            "treatment_median": float(tt.median()),
                            "n_control": int(len(cc)), "n_treatment": int(len(tt))}
        print(f"  {label}: 对照 {cc.median():.0f} 局（n={len(cc)}） vs 实验 {tt.median():.0f} 局（n={len(tt)}）")

    plot_ecdf_rounds(c.sum_gamerounds, t.sum_gamerounds, FIGURES / "03_rounds_ecdf.png", x_max=5000)
    plot_bootstrap(med.observed, med.boot_dist, (med.ci_low, med.ci_high), FIGURES / "04_bootstrap_median.png")
    return {
        "retention_1": {"control_rate": d1.p_control, "treatment_rate": d1.p_treatment,
                        "diff_pp": d1.diff * 100, "ci_pp": [d1.ci_low * 100, d1.ci_high * 100],
                        "p_value": d1.p_value},
        "rounds_median_diff": {"observed": med.observed, "ci": [med.ci_low, med.ci_high]},
        "rounds_mean_ttest_p": float(ttest.pvalue),
        "quantiles": {str(qq): {"control": float(quant.loc[qq, CONTROL]),
                                "treatment": float(quant.loc[qq, TREATMENT])} for qq in q},
        "outlier_sensitivity": sensitivity,
    }


def section_stratified(df: pd.DataFrame) -> dict:
    hr("4. 分层：按累计游戏局数分桶")
    bins = [-1, 0, 5, 20, 100, np.inf]
    labels = ["0", "1-5", "6-20", "21-100", "100+"]
    df = df.copy()
    df["bin"] = pd.cut(df.sum_gamerounds, bins=bins, labels=labels)

    rows = []
    for b in labels:
        sub = df[df.bin == b]
        c, t = sub[sub.version == CONTROL], sub[sub.version == TREATMENT]
        # 小桶里有人可能一组为 0，Wilson 区间能处理，但差值没法算 —— 记录样本量供读者判断
        rc = c.retention_7.mean() if len(c) else np.nan
        rt = t.retention_7.mean() if len(t) else np.nan
        lc, hc = wilson_interval(int(c.retention_7.sum()), len(c)) if len(c) else (np.nan, np.nan)
        lt, ht = wilson_interval(int(t.retention_7.sum()), len(t)) if len(t) else (np.nan, np.nan)
        rows.append({"bin": b, "n_control": len(c), "n_treatment": len(t),
                     "control_rate": rc, "treatment_rate": rt,
                     "control_lo": lc, "control_hi": hc, "treatment_lo": lt, "treatment_hi": ht,
                     "diff_pp": (rc - rt) * 100})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    # 按对照组的局数分布做标准化：回答"如果两组玩家结构一样，留存差多少"
    w = table["n_control"] / table["n_control"].sum()
    std_c = float((table.control_rate * w).sum())
    std_t = float((table.treatment_rate * w).sum())
    print(f"\n按对照组结构标准化：对照 {std_c:.2%} vs 实验 {std_t:.2%}，差 {(std_c - std_t) * 100:+.3f} pp")

    # 交互检验：version 的效应在桶间是否一致（逻辑回归 + 似然比检验）
    import statsmodels.formula.api as smf

    d = df.rename(columns={"version": "grp"})
    main = smf.logit("retention_7 ~ C(grp) + C(bin)", data=d).fit(disp=0)
    inter = smf.logit("retention_7 ~ C(grp) * C(bin)", data=d).fit(disp=0)
    lr = 2 * (inter.llf - main.llf)
    df_diff = inter.df_model - main.df_model
    p_inter = float(stats.chi2.sf(lr, df_diff))
    print(f"交互项似然比检验：LR={lr:.2f}, df={int(df_diff)}, p={p_inter:.3f} "
          f"→ {'各桶效应不一致，需谨慎' if p_inter < ALPHA else '没有证据说明各桶效应不同'}")

    plot_stratified(table, FIGURES / "05_stratified_retention.png",
                    title="分桶后差异集中在 21-100 桶（+2.1pp），其余桶接近 0")
    return {
        "table": table.round(6).to_dict(orient="records"),
        "standardized": {"control": std_c, "treatment": std_t, "diff_pp": (std_c - std_t) * 100},
        "interaction_p": p_inter,
    }


def section_power(df: pd.DataFrame, primary: dict) -> dict:
    hr("5. 功效分析：这个实验的设计够不够用")
    n = int(df.version.value_counts().min())
    p_base = primary["treatment_rate"]  # 以实验组为基准算相对提升
    delta = abs(primary["diff_pp"]) / 100

    power_now = achieved_power(p_base, delta, n)
    mde = min_detectable_delta(p_base, n)
    print(f"每组 n={n:,}，基准留存 {p_base:.2%}")
    print(f"观测效应 {delta * 100:.2f}pp 时，检验功效仅 {power_now:.1%}")
    print(f"80% 功效下能检出的最小提升（MDE）= {mde * 100:.2f}pp")

    plan = {}
    for target in (0.005, 0.01, 0.02):
        need = required_sample_size(p_base, target, alpha=ALPHA, power=0.8)
        plan[f"{target * 100:.1f}pp"] = int(round(need))
        print(f"想稳定检出 {target * 100:.1f}pp 提升，每组需要 {int(round(need)):,} 人（当前 {n:,}）")

    deltas = np.linspace(0.001, 0.03, 60)
    plot_power_curve(p_base, n, deltas, power_now, delta, FIGURES / "06_power_curve.png")
    return {"n_per_group": n, "base_rate": p_base, "observed_delta_pp": delta * 100,
            "achieved_power": float(power_now), "mde_pp": float(mde * 100),
            "required_n": plan}


def section_multiplicity(secondary: dict, primary: dict) -> dict:
    hr("6. 多重比较：次要指标一起看时要打折")
    pvals = [primary["p_value"], secondary["retention_1"]["p_value"]]
    # 局数没有 p 值（bootstrap 区间），这里只对"有 p 值"的指标做校正，并在 README 说明
    names = ["retention_7", "retention_1"]
    bon = bonferroni(pvals)
    for name, p, rej in zip(names, pvals, bon.reject):
        print(f"  {name}: p={p:.4f} → 校正后阈值 {bon.alpha_adjusted:.4f}，{'仍显著' if rej else '不再显著'}")
    return {"names": names, "p_values": pvals, "alpha_adjusted": bon.alpha_adjusted,
            "reject": list(bon.reject)}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()

    results = {}
    results["health"] = section_health(df)
    results["primary"] = section_primary(df)
    results["secondary"] = section_secondary(df)
    results["stratified"] = section_stratified(df)
    results["power"] = section_power(df, results["primary"])
    results["multiplicity"] = section_multiplicity(results["secondary"], results["primary"])

    out = REPORTS / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    print(f"图已写入 {FIGURES}")


if __name__ == "__main__":
    main()
