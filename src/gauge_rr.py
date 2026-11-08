"""
Step 2: Measurement System Analysis (crossed Gauge R&R).

Generates a classic AIAG crossed study (10 wafers x 3 operators x 3 trials,
randomized run order) for three gauges: ellipsometer thickness (center site),
ellipsometer refractive index (center site) and the stress gauge.

Writes:
  data/gauge_rr_study.csv          -> open this in JMP
  data/gauge_rr_results.csv        -> Python ANOVA results to check JMP against
  plots/gauge_rr_variability.png   -> variability chart

Method: two-way ANOVA with Wafer x Operator interaction (AIAG MSA 4th ed.),
variance components from expected mean squares with negatives set to zero,
matching JMP's Variability Chart Gauge R&R. Study variation uses 6 sigma.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FACTORS
from pecvd import Recipe, make_fleet
from metrology import Ellipsometer, StressGauge

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
N_WAFERS, OPERATORS, N_TRIALS = 10, ["OP1", "OP2", "OP3"], 3

# Illustrative spec limits (LSL, USL) used for % tolerance
SPECS = {
    "Thickness_A": (2850.0, 3150.0),
    "RI":          (1.980, 2.020),
    "Stress_MPa":  (-200.0, -100.0),
}
CENTER_SITE = 0


def make_parts(rng):
    """10 production wafers spread across the 4 chambers with normal
    recipe-to-recipe variation, so parts represent the real process range."""
    fleet = make_fleet(seed=SEED)
    names = list(fleet)
    wafers = []
    for i in range(N_WAFERS):
        ch = fleet[names[i % len(names)]]
        r = Recipe(**{k: FACTORS[k][1] * (1 + rng.normal(0, 0.01)) for k in FACTORS})
        w = ch.run(r)
        w.wafer_id = f"W{i + 1:02d}"
        wafers.append(w)
    return wafers


def generate_study():
    rng = np.random.default_rng(SEED)
    wafers = make_parts(rng)
    ellip = Ellipsometer(np.random.default_rng(SEED + 1))
    sg = StressGauge(np.random.default_rng(SEED + 2))

    rows = []
    for trial in range(1, N_TRIALS + 1):
        for op in OPERATORS:
            for w in wafers:
                thk, ri = ellip.measure(w, op)
                rows.append({
                    "Wafer": w.wafer_id, "Operator": op, "Trial": trial,
                    "Thickness_A": round(float(thk[CENTER_SITE]), 2),
                    "RI": round(float(ri[CENTER_SITE]), 5),
                    "Stress_MPa": round(float(sg.measure(w, op)), 2),
                })
    df = pd.DataFrame(rows)
    # Randomized run order, as a real study would be run
    order = rng.permutation(len(df)) + 1
    df.insert(0, "RunOrder", order)
    return df.sort_values("RunOrder").reset_index(drop=True)


def anova_grr(df, y, lsl, usl, alpha_pool=None, k=6.0):
    """AIAG ANOVA-method Gauge R&R for a balanced crossed study.
    alpha_pool=None keeps the interaction term (matches JMP's default);
    alpha_pool=0.25 pools a non-significant interaction (AIAG option)."""
    p, o = df["Wafer"].nunique(), df["Operator"].nunique()
    r = len(df) // (p * o)
    g = df[y].mean()
    m_part = df.groupby("Wafer")[y].mean()
    m_op = df.groupby("Operator")[y].mean()
    m_cell = df.groupby(["Wafer", "Operator"])[y].mean()

    ss_part = o * r * ((m_part - g) ** 2).sum()
    ss_op = p * r * ((m_op - g) ** 2).sum()
    cell = m_cell.reset_index()
    inter = cell[y] - cell["Wafer"].map(m_part) - cell["Operator"].map(m_op) + g
    ss_int = r * (inter ** 2).sum()
    ss_err = ((df[y] - df.set_index(["Wafer", "Operator"]).index.map(m_cell)) ** 2).sum()

    df_part, df_op, df_int, df_err = p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1)
    ms_part, ms_op = ss_part / df_part, ss_op / df_op
    ms_int, ms_err = ss_int / df_int, ss_err / df_err
    p_int = stats.f.sf(ms_int / ms_err, df_int, df_err)

    pooled = alpha_pool is not None and p_int > alpha_pool
    if pooled:
        ms_e = (ss_int + ss_err) / (df_int + df_err)
        var_rep, var_int, denom = ms_e, 0.0, ms_e
    else:
        var_rep = ms_err
        var_int = max(0.0, (ms_int - ms_err) / r)
        denom = ms_int
    var_op = max(0.0, (ms_op - denom) / (p * r))
    var_part = max(0.0, (ms_part - denom) / (o * r))

    var_repro = var_op + var_int
    var_grr = var_rep + var_repro
    var_total = var_grr + var_part
    tol = usl - lsl

    def pct_sv(v):  # % of study variation (ratio of std devs)
        return 100 * np.sqrt(v / var_total)

    def pct_tol(v):
        return 100 * k * np.sqrt(v) / tol

    res = {
        "Response": y,
        "Interaction p": round(p_int, 3),
        "Interaction pooled": pooled,
        "Repeatability SD": np.sqrt(var_rep),
        "Reproducibility SD": np.sqrt(var_repro),
        "GRR SD": np.sqrt(var_grr),
        "Part SD": np.sqrt(var_part),
        "%SV Repeatability": pct_sv(var_rep),
        "%SV Reproducibility": pct_sv(var_repro),
        "%SV GRR": pct_sv(var_grr),
        "%SV Part": pct_sv(var_part),
        "%Tol GRR": pct_tol(var_grr),
        "ndc": int(np.floor(1.41 * np.sqrt(var_part / var_grr))) if var_grr > 0 else 999,
    }
    worst = max(res["%SV GRR"], res["%Tol GRR"])
    res["Verdict"] = ("Acceptable" if worst < 10 else
                      "Marginal" if worst <= 30 else "Not acceptable")
    return res


def plot_variability(df, path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    colors = {"OP1": "#2a6f97", "OP2": "#e07a1f", "OP3": "#3a9d5d"}
    wafers = sorted(df["Wafer"].unique())
    for ax, (y, (lsl, usl)) in zip(axes, SPECS.items()):
        for i, w in enumerate(wafers):
            for j, op in enumerate(colors):
                vals = df[(df.Wafer == w) & (df.Operator == op)][y]
                x = i * 4 + j
                ax.scatter([x] * len(vals), vals, color=colors[op], s=18,
                           label=op if i == 0 else None)
                ax.plot([x - 0.3, x + 0.3], [vals.mean()] * 2, color=colors[op], lw=2)
        ax.set_xticks([i * 4 + 1 for i in range(len(wafers))], wafers)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=3, loc="upper right")
    fig.suptitle("Gauge R&R variability chart: 10 wafers x 3 operators x 3 trials")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    df = generate_study()
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "gauge_rr_study.csv", index=False)

    results = pd.DataFrame([anova_grr(df, y, *SPECS[y]) for y in SPECS])
    results.to_csv(ROOT / "data" / "gauge_rr_results.csv", index=False, float_format="%.5g")
    plot_variability(df, ROOT / "plots" / "gauge_rr_variability.png")

    pd.set_option("display.width", 200)
    show = results.set_index("Response").T
    print(show.to_string(float_format=lambda v: f"{v:.4g}"))
    print("\nAIAG guideline: <10% acceptable, 10-30% marginal, >30% not acceptable; ndc >= 5.")


if __name__ == "__main__":
    main()
