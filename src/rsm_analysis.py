"""
Step 4: Response surface model + multi-response optimization.

Fits a full quadratic model (main effects, 2-factor interactions, squared
terms) to each response of the augmented central composite design, the same
model as JMP Fit Model > Macros > Response Surface. If the design has a Block
column, the block effect is tested and dropped when not significant.

Then finds the recipe with the best overall desirability (Derringer-Suich):
  Dep_Rate  maximize     (40 -> 80 nm/min)
  WIW_NU    minimize     (2.0 -> 0.5 %)
  RI        target 2.00  (1.98 - 2.02)
  Stress    target -150  (-200 - -100 MPa)

Usage (from the src folder):
    python rsm_analysis.py ..\\data\\rsm_results.csv
    python rsm_analysis.py ..\\data\\rsm_results.csv --at "RF_Power=520,Pressure=4.2,Temp=330,NH3_SiH4=4.1,Spacing=380"
The --at option prints model predictions at a recipe (e.g. JMP's optimum) so
they can be compared with JMP's Prediction Profiler.
"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_doe import map_factor_columns, RESPONSES

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["rf_power", "pressure", "temp", "nh3_sih4", "spacing"]
SHORT = {"rf_power": "Power", "pressure": "Press", "temp": "Temp",
         "nh3_sih4": "NH3/SiH4", "spacing": "Spacing"}

DESIRABILITY = {
    "Dep_Rate": ("max", 40.0, 80.0),
    "WIW_NU":   ("min", 0.5, 2.0),
    "RI":       ("target", 1.98, 2.00, 2.02),
    "Stress":   ("target", -200.0, -150.0, -100.0),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Coder:
    """Actual <-> coded units, using the design's low/high for each factor."""
    def __init__(self, df):
        self.cols = map_factor_columns(df)
        self.lo = np.array([df[self.cols[f]].min() for f in ORDER], float)
        self.hi = np.array([df[self.cols[f]].max() for f in ORDER], float)
        self.mid, self.half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2

    def code(self, actual):
        return (np.asarray(actual, float) - self.mid) / self.half

    def decode(self, coded):
        return self.mid + np.asarray(coded, float) * self.half

    def coded_frame(self, df):
        return np.column_stack([(df[self.cols[f]].to_numpy(float) - m) / h
                                for f, m, h in zip(ORDER, self.mid, self.half)])


def term_names():
    n = [SHORT[f] for f in ORDER]
    return n + [f"{a}*{b}" for a, b in combinations(n, 2)] + [f"{a}^2" for a in n]


def quad_terms(Xc):
    Xc = np.atleast_2d(Xc)
    k = Xc.shape[1]
    inter = [Xc[:, i] * Xc[:, j] for i, j in combinations(range(k), 2)]
    sq = [Xc[:, i] ** 2 for i in range(k)]
    return np.column_stack([Xc] + inter + sq)


class QuadModel:
    def __init__(self, Xc, y, block=None):
        cols = [np.ones(len(y)), quad_terms(Xc)]
        self.names = ["Intercept"] + term_names()
        self.block_levels = None
        if block is not None:
            levels = sorted(pd.unique(block))
            self.block_levels = levels
            # effect (sum-to-zero) coding, like JMP: predictions average over blocks
            for lv in levels[:-1]:
                cols.append(np.where(block == lv, 1.0, np.where(block == levels[-1], -1.0, 0.0)))
                self.names.append(f"Block[{lv}]")
        A = np.column_stack(cols)
        self.A, self.y = A, y
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ self.beta
        self.dof = len(y) - A.shape[1]
        self.sse = resid @ resid
        self.s2 = self.sse / self.dof
        self.XtX_inv = np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.diag(self.XtX_inv) * self.s2)
        self.t = self.beta / se
        self.p = 2 * stats.t.sf(np.abs(self.t), self.dof)
        sst = ((y - y.mean()) ** 2).sum()
        self.r2 = 1 - self.sse / sst
        self.r2_adj = 1 - (self.sse / self.dof) / (sst / (len(y) - 1))
        self.rmse = np.sqrt(self.s2)

    def row(self, Xc):
        r = np.column_stack([np.ones(len(np.atleast_2d(Xc))), quad_terms(Xc)])
        if self.block_levels:
            r = np.column_stack([r, np.zeros((len(r), len(self.block_levels) - 1))])
        return r

    def predict(self, Xc):
        return self.row(Xc) @ self.beta

    def pred_interval(self, Xc, alpha=0.05):
        r = self.row(Xc)
        se = np.sqrt(self.s2 * (1 + np.einsum("ij,jk,ik->i", r, self.XtX_inv, r)))
        t = stats.t.ppf(1 - alpha / 2, self.dof)
        yhat = r @ self.beta
        return yhat - t * se, yhat + t * se

    def block_p(self):
        return [p for n, p in zip(self.names, self.p) if n.startswith("Block")]


def lack_of_fit(Xc, y, model):
    """Pure error from replicated factor settings (center points)."""
    keys = pd.Series(["|".join(f"{v:.6f}" for v in r) for r in Xc])
    groups = pd.Series(y).groupby(keys)
    ss_pe = sum(((g - g.mean()) ** 2).sum() for _, g in groups)
    df_pe = sum(len(g) - 1 for _, g in groups)
    df_lof = model.dof - df_pe
    if df_pe < 1 or df_lof < 1:
        return None
    f = ((model.sse - ss_pe) / df_lof) / (ss_pe / df_pe)
    return stats.f.sf(f, df_lof, df_pe)


def fit_all(df, verbose=True):
    coder = Coder(df)
    Xc = coder.coded_frame(df)
    block = df["Block"].to_numpy() if "Block" in df.columns and df["Block"].nunique() > 1 else None
    models = {}
    if block is not None:
        trial = {r: QuadModel(Xc, df[r].to_numpy(float), block) for r in RESPONSES}
        bp = {r: min(m.block_p()) for r, m in trial.items()}
        keep = any(p < 0.05 for p in bp.values())
        if verbose:
            print("Block effect p-values: " + ", ".join(f"{r} {p:.3f}" for r, p in bp.items()))
            print("-> block is significant, kept in model" if keep else
                  "-> block not significant for any response, dropped from model")
        if not keep:
            block = None
    for r in RESPONSES:
        models[r] = QuadModel(Xc, df[r].to_numpy(float), block)
    return coder, Xc, models


# ---------------------------------------------------------------------------
# Desirability
# ---------------------------------------------------------------------------
def desirability(resp, y):
    spec = DESIRABILITY[resp]
    if spec[0] == "max":
        _, lo, hi = spec
        return np.clip((y - lo) / (hi - lo), 0, 1)
    if spec[0] == "min":
        _, best, worst = spec
        return np.clip((worst - y) / (worst - best), 0, 1)
    _, lo, tgt, hi = spec
    return np.clip(np.where(y <= tgt, (y - lo) / (tgt - lo), (hi - y) / (hi - tgt)), 0, 1)


def overall_d(models, Xc):
    ds = [desirability(r, models[r].predict(Xc)) for r in RESPONSES]
    return np.prod(ds, axis=0) ** (1 / len(ds))


def optimize_recipe(models, n_start=200, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for x0 in rng.uniform(-1, 1, size=(n_start, 5)):
        res = optimize.minimize(lambda x: -overall_d(models, x[None, :])[0], x0,
                                bounds=[(-1, 1)] * 5, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def contour_plots(coder, models, x_opt, path):
    panels = [("Dep_Rate", 0, 3), ("RI", 0, 3), ("WIW_NU", 0, 4), ("Stress", 1, 2)]
    labels = ["RF power (W)", "Pressure (Torr)", "Temp (C)", "NH3/SiH4", "Spacing (mils)"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    g = np.linspace(-1, 1, 81)
    for ax, (resp, i, j) in zip(axes, panels):
        G1, G2 = np.meshgrid(g, g)
        X = np.tile(x_opt, (G1.size, 1))
        X[:, i], X[:, j] = G1.ravel(), G2.ravel()
        Z = models[resp].predict(X).reshape(G1.shape)
        a1 = coder.decode(np.column_stack([g] * 5))[:, i]
        a2 = coder.decode(np.column_stack([g] * 5))[:, j]
        cs = ax.contourf(a1, a2, Z, levels=14, cmap="viridis")
        ax.contour(a1, a2, Z, levels=14, colors="white", linewidths=0.4)
        fig.colorbar(cs, ax=ax)
        opt = coder.decode(x_opt)
        ax.plot(opt[i], opt[j], marker="*", color="red", ms=16, mec="white")
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.set_title(resp)
    fig.suptitle("Response surfaces through the optimum recipe (red star); other factors held at optimum")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ---------------------------------------------------------------------------
def parse_at(text, coder):
    vals = dict(kv.split("=") for kv in text.split(","))
    lookup = {c.lower(): f for f, c in coder.cols.items()}
    actual = {lookup[k.strip().lower()]: float(v) for k, v in vals.items()}
    return coder.code([actual[f] for f in ORDER])


def print_recipe(coder, models, xc, title):
    actual = coder.decode(xc)
    print(f"\n{title}")
    for f, v in zip(ORDER, actual):
        print(f"  {coder.cols[f]:9s} = {v:8.2f}")
    for r in RESPONSES:
        y = models[r].predict(xc[None, :])[0]
        lo, hi = models[r].pred_interval(xc[None, :])
        print(f"  {r:9s} predicted {y:9.4g}   95% PI [{lo[0]:.4g}, {hi[0]:.4g}]"
              f"   desirability {desirability(r, y):.3f}")
    print(f"  Overall desirability: {overall_d(models, xc[None, :])[0]:.3f}")


def main():
    args = sys.argv[1:]
    at = None
    if "--at" in args:
        k = args.index("--at"); at = args[k + 1]; del args[k:k + 2]
    path = Path(args[0]) if args else ROOT / "data" / "rsm_results.csv"
    df = pd.read_csv(path)
    coder, Xc, models = fit_all(df)

    for r in RESPONSES:
        m = models[r]
        lof = lack_of_fit(Xc, df[r].to_numpy(float), m)
        sig = [(n, t) for n, t, p in zip(m.names[1:], m.t[1:], m.p[1:]) if p < 0.05]
        sig.sort(key=lambda z: -abs(z[1]))
        lof_txt = f"  Lack-of-fit p {lof:.3f}" if lof is not None else ""
        print(f"\n=== {r}: R2 {m.r2:.4f}  R2adj {m.r2_adj:.4f}  RMSE {m.rmse:.4g}{lof_txt}")
        print("  Significant terms (|t| order): " + ", ".join(f"{n} ({t:+.1f})" for n, t in sig))

    x_opt, d = optimize_recipe(models)
    print_recipe(coder, models, x_opt, "Python optimum (Derringer-Suich desirability):")
    if at:
        print_recipe(coder, models, parse_at(at, coder), f"Predictions at --at recipe:")

    out = ROOT / "plots" / "rsm_contours.png"
    contour_plots(coder, models, x_opt, out)
    print(f"\nContour plot saved: {out}")


if __name__ == "__main__":
    main()
