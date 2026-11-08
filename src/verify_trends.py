"""
Check that the simulator reproduces the directions reported in published
PECVD SiNx studies. Sweeps one factor at a time (others at center, no noise),
asserts each direction, and saves plots/one_factor_sweeps.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import FACTORS
from pecvd import Recipe, Chamber

ROOT = Path(__file__).resolve().parents[1]
RESPONSES = [("dep_rate_nm_min", "Dep rate (nm/min)"),
             ("ri", "Refractive index"),
             ("stress_mpa", "Stress (MPa, - = compressive)"),
             ("wiw_nu_pct", "WIW non-uniformity (%)")]

# Expected sign of the change from low to high, from the literature (README).
# None = not checked (no clear published direction).
EXPECTED = {
    "rf_power": {"dep_rate_nm_min": +1, "ri": -1, "stress_mpa": +1, "wiw_nu_pct": +1},
    "pressure": {"dep_rate_nm_min": None, "ri": None, "stress_mpa": +1, "wiw_nu_pct": -1},
    "temp":     {"dep_rate_nm_min": -1, "ri": None, "stress_mpa": -1, "wiw_nu_pct": -1},
    "nh3_sih4": {"dep_rate_nm_min": -1, "ri": -1, "stress_mpa": +1, "wiw_nu_pct": None},
    "spacing":  {"dep_rate_nm_min": -1, "ri": None, "stress_mpa": +1, "wiw_nu_pct": +1},
}


def sweep(factor, n=21):
    lo, c, hi, _ = FACTORS[factor]
    vals = np.geomspace(lo, hi, n) if factor == "nh3_sih4" else np.linspace(lo, hi, n)
    ch = Chamber("T1-A")
    rows = []
    for v in vals:
        r = Recipe(**{factor: v})
        ch.clean()
        rows.append(ch.run(r, noise=False).summary())
    return vals, rows


def main():
    fig, axes = plt.subplots(len(RESPONSES), len(FACTORS), figsize=(16, 11), sharey="row")
    failures = []
    for j, f in enumerate(FACTORS):
        vals, rows = sweep(f)
        for i, (key, label) in enumerate(RESPONSES):
            y = np.array([r[key] for r in rows])
            ax = axes[i, j]
            ax.plot(vals, y, color="#2a6f97", lw=2)
            if f == "nh3_sih4":
                ax.set_xscale("log")
            if i == len(RESPONSES) - 1:
                ax.set_xlabel(f"{f} ({FACTORS[f][3]})")
            if j == 0:
                ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            exp = EXPECTED[f][key]
            if exp is not None:
                got = np.sign(y[-1] - y[0])
                ok = got == exp
                if not ok:
                    failures.append(f"{f} -> {key}: expected {exp:+d}, got {got:+.0f}")
                ax.set_title("trend OK" if ok else "MISMATCH", fontsize=8,
                             color="green" if ok else "red")
    fig.suptitle("Simulated PECVD SiNx: one-factor sweeps (others at center, no noise)")
    fig.tight_layout()
    out = ROOT / "plots" / "one_factor_sweeps.png"
    fig.savefig(out, dpi=130)

    # Print ranges across the full factor space corners for sanity
    print("Center point:", Chamber("T1-A").run(Recipe(), noise=False).summary())
    for f in FACTORS:
        vals, rows = sweep(f, n=3)
        print(f"{f:9s}", [(round(v, 2), r["dep_rate_nm_min"], r["ri"], r["stress_mpa"], r["wiw_nu_pct"])
                          for v, r in zip(vals, rows)])
    if failures:
        print("\nMISMATCHES:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print(f"\nAll {sum(v is not None for d in EXPECTED.values() for v in d.values())} "
          f"literature trend checks passed. Plot: {out}")


if __name__ == "__main__":
    main()
