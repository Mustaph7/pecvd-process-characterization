"""
Step 4: Confirmation runs.

Runs the chosen recipe on a freshly cleaned chamber for several wafers, then
checks each result against the RSM model's 95% prediction interval. If the
wafers land inside the intervals, the model is confirmed and the recipe can
be released.

Usage (from the src folder):
    python confirm.py ..\\data\\rsm_results.csv "RF_Power=520,Pressure=4.2,Temp=330,NH3_SiH4=4.1,Spacing=380"
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from pecvd import Recipe, Chamber
from metrology import Ellipsometer, StressGauge
from run_doe import measure, RESPONSES, CHAMBER
from rsm_analysis import fit_all, parse_at, ORDER

N_WAFERS = 5
SEED = 777
ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    df = pd.read_csv(sys.argv[1])
    coder, _, models = fit_all(df, verbose=False)
    xc = parse_at(sys.argv[2], coder)
    actual = dict(zip(ORDER, coder.decode(xc)))

    rng = np.random.default_rng(SEED)
    chamber = Chamber(CHAMBER, rng)
    chamber.clean()
    ellip, sg = Ellipsometer(rng), StressGauge(rng)
    rows = [measure(chamber.run(Recipe(**actual)), ellip, sg) for _ in range(N_WAFERS)]
    res = pd.DataFrame(rows, index=[f"C{i + 1}" for i in range(N_WAFERS)])

    print(f"Confirmation runs: {N_WAFERS} wafers on {CHAMBER} at "
          + ", ".join(f"{coder.cols[f]}={v:g}" for f, v in actual.items()))
    print(res.to_string())
    summary = []
    for r in RESPONSES:
        yhat = models[r].predict(xc[None, :])[0]
        lo, hi = (v[0] for v in models[r].pred_interval(xc[None, :]))
        inside = res[r].between(lo, hi).sum()
        summary.append({"Response": r, "Predicted": round(yhat, 4),
                        "95% PI low": round(lo, 4), "95% PI high": round(hi, 4),
                        "Confirmation mean": round(res[r].mean(), 4),
                        "Wafers inside PI": f"{inside}/{N_WAFERS}"})
    summary = pd.DataFrame(summary)
    print("\n" + summary.to_string(index=False))
    ok = all(s.split("/")[0] == str(N_WAFERS) for s in summary["Wafers inside PI"])
    print("\nModel CONFIRMED: every wafer inside its prediction interval." if ok else
          "\nSome wafers fell outside the prediction interval: check the model before release.")
    out = ROOT / "data" / "confirmation_runs.csv"
    res.to_csv(out)
    summary.to_csv(ROOT / "data" / "confirmation_summary.csv", index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
