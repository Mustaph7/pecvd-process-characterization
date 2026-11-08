"""
Step 5: Fleet variance components and chamber matching.

Runs the released recipe (Step 4) on all 4 chambers:
    4 chambers x 5 days x 3 wafers x 49 sites
Each day starts with a chamber clean and a small random day-to-day shift.

  python fleet_study.py baseline
      Same deposition time on every chamber. Writes site- and wafer-level
      data for JMP, nested variance components, chamber ANOVA + Tukey,
      equivalence tests, Ppk per chamber, and the per-chamber deposition
      time adjustments that would match the fleet.

  python fleet_study.py matched
      Repeats the study with the per-chamber deposition times from
      data/chamber_dep_times.csv and compares before vs after.

Thickness spec (wafer mean): 3000 A +/- 2.5%  ->  2925 - 3075 A
"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CHAMBERS
from pecvd import Recipe, Chamber, mean_dep_rate, SITE_X, SITE_Y
from metrology import Ellipsometer, StressGauge

ROOT = Path(__file__).resolve().parents[1]
RECIPE = Recipe(rf_power=600, pressure=4.67, temp=400, nh3_sih4=2.92, spacing=300)
TARGET_A, LSL_A, USL_A = 3000.0, 2925.0, 3075.0
EQUIV_MARGIN_A = 30.0          # chambers "equivalent" if means within +/-1% (30 A)
N_DAYS, N_WAFERS = 5, 3
SEEDS = {"baseline": 500, "matched": 501}
STRESS_REPEATS = 3
TIMES_FILE = ROOT / "data" / "chamber_dep_times.csv"


def base_dep_time():
    """Deposition time that gives 3000 A on a nominal chamber."""
    return TARGET_A / (mean_dep_rate(RECIPE) * 10 / 60)


def run_fleet(mode):
    if mode == "matched":
        times = pd.read_csv(TIMES_FILE).set_index("Chamber")["Dep_Time_s"].to_dict()
    else:
        times = {c: round(base_dep_time(), 1) for c in CHAMBERS}
    rng = np.random.default_rng(SEEDS[mode])
    ellip, sg = Ellipsometer(rng), StressGauge(rng)
    sites, wafers = [], []
    for ch_name in CHAMBERS:
        ch = Chamber(ch_name, rng)
        for day in range(1, N_DAYS + 1):
            ch.new_day()
            for w in range(1, N_WAFERS + 1):
                wafer = ch.run(RECIPE, dep_time_s=times[ch_name])
                thk, ri = ellip.measure(wafer, "OP1")
                wid = f"{ch_name}-D{day}-W{w}"
                for k in range(thk.size):
                    sites.append({"Chamber": ch_name, "Day": day, "Wafer": w, "Wafer_ID": wid,
                                  "Site": k + 1, "X_mm": round(SITE_X[k], 1), "Y_mm": round(SITE_Y[k], 1),
                                  "Thickness_A": round(thk[k], 2), "RI": round(ri[k], 5)})
                wafers.append({"Chamber": ch_name, "Day": day, "Wafer": w, "Wafer_ID": wid,
                               "Dep_Time_s": times[ch_name],
                               "Mean_Thk_A": round(thk.mean(), 2),
                               "WIW_NU": round(thk.std(ddof=1) / thk.mean() * 100, 3),
                               "RI": round(ri.mean(), 5),
                               "Stress": round(np.mean([sg.measure(wafer, "OP1")
                                                        for _ in range(STRESS_REPEATS)]), 1)})
    return pd.DataFrame(sites), pd.DataFrame(wafers)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def nested_vc(df, y):
    """Balanced nested ANOVA: Chamber > Day > Wafer > Site (within wafer).
    Variance components from expected mean squares, negatives set to 0."""
    a = df["Chamber"].nunique(); b = df.groupby("Chamber")["Day"].nunique().iloc[0]
    c = df.groupby(["Chamber", "Day"])["Wafer"].nunique().iloc[0]; n = len(df) // (a * b * c)
    g = df[y].mean()
    mA = df.groupby("Chamber")[y].transform("mean")
    mB = df.groupby(["Chamber", "Day"])[y].transform("mean")
    mC = df.groupby(["Chamber", "Day", "Wafer"])[y].transform("mean")
    ss = {"Chamber": ((mA - g) ** 2).sum(), "Day": ((mB - mA) ** 2).sum(),
          "Wafer": ((mC - mB) ** 2).sum(), "Site": ((df[y] - mC) ** 2).sum()}
    dof = {"Chamber": a - 1, "Day": a * (b - 1), "Wafer": a * b * (c - 1), "Site": a * b * c * (n - 1)}
    ms = {k: ss[k] / dof[k] for k in ss}
    vc = {"Site": ms["Site"],
          "Wafer": max(0.0, (ms["Wafer"] - ms["Site"]) / n),
          "Day": max(0.0, (ms["Day"] - ms["Wafer"]) / (c * n)),
          "Chamber": max(0.0, (ms["Chamber"] - ms["Day"]) / (b * c * n))}
    total = sum(vc.values())
    out = pd.DataFrame({"Var Component": vc, "Std Dev": {k: np.sqrt(v) for k, v in vc.items()},
                        "% of Total": {k: 100 * v / total for k, v in vc.items()}})
    out = out.loc[["Chamber", "Day", "Wafer", "Site"]]
    out.loc["Total"] = [total, np.sqrt(total), 100.0]
    return out


def chamber_tests(wdf, y, margin):
    groups = [g[y].to_numpy() for _, g in wdf.groupby("Chamber")]
    names = sorted(wdf["Chamber"].unique())
    f, p = stats.f_oneway(*groups)
    tk = stats.tukey_hsd(*groups)
    k, N = len(groups), sum(len(g) for g in groups)
    s_p = np.sqrt(sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k))
    rows = []
    for (i, a), (j, b) in combinations(enumerate(names), 2):
        d = groups[i].mean() - groups[j].mean()
        se = s_p * np.sqrt(1 / len(groups[i]) + 1 / len(groups[j]))
        p_tost = max(stats.t.sf((d + margin) / se, N - k), stats.t.cdf((d - margin) / se, N - k))
        rows.append({"Pair": f"{a} - {b}", "Difference": round(d, 2),
                     "Tukey p": round(tk.pvalue[i, j], 4),
                     "Equivalence p": round(p_tost, 4),
                     "Equivalent (+/-%g)" % margin: "Yes" if p_tost < 0.05 else "No"})
    return f, p, pd.DataFrame(rows)


def ppk(x, lsl, usl):
    m, s = np.mean(x), np.std(x, ddof=1)
    return min(usl - m, m - lsl) / (3 * s)


def capability(wdf):
    rows = []
    for ch, g in list(wdf.groupby("Chamber")) + [("Fleet", wdf)]:
        x = g["Mean_Thk_A"]
        rows.append({"Chamber": ch, "Mean": round(x.mean(), 1), "Std Dev": round(x.std(ddof=1), 2),
                     "Ppk": round(ppk(x, LSL_A, USL_A), 2),
                     "Out of spec": int(((x < LSL_A) | (x > USL_A)).sum())})
    return pd.DataFrame(rows).set_index("Chamber")


# ---------------------------------------------------------------------------
def report(mode, sdf, wdf):
    pd.set_option("display.width", 160)
    print(f"\n######## {mode.upper()}: {len(wdf)} wafers, {len(sdf)} site measurements ########")
    print("\nNested variance components, Thickness_A (site level):")
    vc = nested_vc(sdf, "Thickness_A")
    print(vc.round(3).to_string())
    print("\nNested variance components, RI (site level):")
    print(nested_vc(sdf, "RI").map(lambda v: f"{v:.3g}").to_string())
    f, p, pairs = chamber_tests(wdf, "Mean_Thk_A", EQUIV_MARGIN_A)
    print(f"\nChamber ANOVA on wafer mean thickness: F = {f:.1f}, p = {p:.2g}")
    print(pairs.to_string(index=False))
    for y in ["RI", "Stress", "WIW_NU"]:
        f2, p2 = stats.f_oneway(*[g[y] for _, g in wdf.groupby("Chamber")])
        means = wdf.groupby("Chamber")[y].mean().round(4).to_dict()
        print(f"Chamber ANOVA {y}: p = {p2:.2g}   means {means}")
    cap = capability(wdf)
    print(f"\nThickness capability vs spec {LSL_A:g}-{USL_A:g} A (overall sigma, Ppk):")
    print(cap.to_string())
    return vc, cap


def plot(sb, wb, sm, wm, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    names = list(CHAMBERS)
    for ax, (w, title) in zip(axes[:2], [(wb, "Before matching"), (wm, "After matching")]):
        data = [w[w.Chamber == c]["Mean_Thk_A"] for c in names]
        ax.boxplot(data, tick_labels=names)
        for i, d in enumerate(data, 1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(d)), d, s=12, color="#2a6f97", zorder=3)
        for v, ls in [(LSL_A, "--"), (USL_A, "--"), (TARGET_A, ":")]:
            ax.axhline(v, color="red" if ls == "--" else "gray", ls=ls, lw=1)
        ax.set_ylim(2900, 3100); ax.set_title(title); ax.set_ylabel("Wafer mean thickness (A)")
    vb, vm = nested_vc(sb, "Thickness_A"), nested_vc(sm, "Thickness_A")
    comps = ["Chamber", "Day", "Wafer", "Site"]
    x = np.arange(len(comps))
    axes[2].bar(x - 0.2, vb.loc[comps, "Std Dev"], 0.4, label="Before", color="#b8c4ce")
    axes[2].bar(x + 0.2, vm.loc[comps, "Std Dev"], 0.4, label="After", color="#2a6f97")
    axes[2].set_xticks(x, comps); axes[2].set_ylabel("Std dev (A)")
    axes[2].set_title("Thickness variance components"); axes[2].legend()
    fig.suptitle("Chamber matching by per-chamber deposition time (spec limits dashed red)")
    fig.tight_layout(); fig.savefig(path, dpi=130)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if mode not in SEEDS:
        raise SystemExit("Usage: python fleet_study.py baseline|matched")
    sdf, wdf = run_fleet(mode)
    sdf.to_csv(ROOT / "data" / f"fleet_{mode}_sites.csv", index=False)
    wdf.to_csv(ROOT / "data" / f"fleet_{mode}_wafers.csv", index=False)
    vc, cap = report(mode, sdf, wdf)

    if mode == "baseline":
        base = base_dep_time()
        times = pd.DataFrame({"Chamber": cap.index[:-1],
                              "Measured_Mean_A": cap["Mean"].iloc[:-1].values})
        times["Dep_Time_s"] = (round(base, 1) * TARGET_A / times["Measured_Mean_A"]).round(1)
        times["Change_s"] = (times["Dep_Time_s"] - round(base, 1)).round(1)
        times.to_csv(TIMES_FILE, index=False)
        print(f"\nBase deposition time {base:.1f} s. Per-chamber times to center each chamber on {TARGET_A:g} A:")
        print(times.to_string(index=False))
        print(f"Saved: {TIMES_FILE}\nNext: python fleet_study.py matched")
    else:
        sb = pd.read_csv(ROOT / "data" / "fleet_baseline_sites.csv")
        wb = pd.read_csv(ROOT / "data" / "fleet_baseline_wafers.csv")
        vb, cb = nested_vc(sb, "Thickness_A"), capability(wb)
        print("\nBEFORE vs AFTER")
        comp = pd.DataFrame({"Chamber SD before": vb["Std Dev"], "Chamber SD after": vc["Std Dev"]}).round(2)
        print(comp.to_string())
        print(pd.DataFrame({"Ppk before": cb["Ppk"], "Ppk after": cap["Ppk"]}).to_string())
        out = ROOT / "plots" / "chamber_matching.png"
        plot(sb, wb, sdf, wdf, out)
        print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
