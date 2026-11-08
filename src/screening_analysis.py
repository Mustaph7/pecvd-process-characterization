"""
Step 3 check: analyze the screening results in Python to verify JMP's Fit Model.

Model: main effects + all two-factor interactions in coded units (-1..+1),
the same model JMP fits with Macros > Factorial to Degree (2).
Prints t-ratios and p-values for each response (compare with JMP's
"Sorted Parameter Estimates"), a curvature test from the center points,
and saves a Pareto chart of effects.

Usage (from the src folder):
    python screening_analysis.py ..\\data\\screening_results.csv
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

from run_doe import map_factor_columns, RESPONSES

ROOT = Path(__file__).resolve().parents[1]
SHORT = {"rf_power": "Power", "pressure": "Press", "temp": "Temp",
         "nh3_sih4": "NH3/SiH4", "spacing": "Spacing"}


def coded_matrix(df):
    cols = map_factor_columns(df)
    X = {}
    for f, c in cols.items():
        lo, hi = df[c].min(), df[c].max()
        X[SHORT[f]] = (df[c] - (lo + hi) / 2) / ((hi - lo) / 2)
    X = pd.DataFrame(X)
    for a, b in combinations(list(X.columns), 2):
        X[f"{a}*{b}"] = X[a] * X[b]
    return X


def fit(X, y):
    A = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * s2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    names = ["Intercept"] + list(X.columns)
    return pd.DataFrame({"Estimate": beta, "Std Error": se, "t Ratio": t, "p": p},
                        index=names), dof, np.sqrt(s2)


def curvature_test(X, y):
    """Mean of factorial points vs mean of center points."""
    main = X.iloc[:, :5]
    center = (main.abs() < 1e-9).all(axis=1)
    if center.sum() < 2:
        return None
    yf, yc = y[~center], y[center]
    diff = yf.mean() - yc.mean()
    se = yc.std(ddof=1) * np.sqrt(1 / len(yf) + 1 / len(yc))
    t = diff / se
    return diff, 2 * stats.t.sf(abs(t), len(yc) - 1)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "screening_results.csv"
    df = pd.read_csv(path)
    X = coded_matrix(df)
    pd.set_option("display.width", 160)

    fig, axes = plt.subplots(1, len(RESPONSES), figsize=(18, 6))
    for ax, resp in zip(axes, RESPONSES):
        y = df[resp].to_numpy(float)
        table, dof, rmse = fit(X, y)
        terms = table.drop("Intercept").sort_values("t Ratio", key=np.abs, ascending=False)
        sig = terms[terms["p"] < 0.05]
        print(f"\n=== {resp}  (RMSE {rmse:.4g}, error df {dof}) ===")
        print(terms.round(4).to_string())
        print(f"Significant at 0.05: {', '.join(sig.index) if len(sig) else 'none'}")
        c = curvature_test(X, y)
        if c:
            print(f"Curvature (factorial mean - center mean): {c[0]:.4g}, p = {c[1]:.4f}"
                  f"{'  -> curvature present, RSM needed' if c[1] < 0.05 else ''}")

        top = terms.iloc[:10][::-1]
        colors = ["#2a6f97" if p < 0.05 else "#b8c4ce" for p in top["p"]]
        ax.barh(top.index, top["t Ratio"].abs(), color=colors)
        ax.axvline(stats.t.ppf(0.975, dof), color="red", ls="--", lw=1)
        ax.set_title(resp)
        ax.set_xlabel("|t Ratio|")
    fig.suptitle("Screening DOE: Pareto of effects (blue = significant at 0.05, red line = t critical)")
    fig.tight_layout()
    out = ROOT / "plots" / "screening_pareto.png"
    fig.savefig(out, dpi=130)
    print(f"\nPareto chart saved: {out}")


if __name__ == "__main__":
    main()
