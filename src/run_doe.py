"""
"Run" a DOE on the simulated PECVD chamber.

Takes a design table exported from JMP (CSV), processes one wafer per row on
chamber T1-A in the row order of the file (JMP's randomized run order), measures
it, and writes the table back out with the responses filled in.

Usage (from the src folder):
    python run_doe.py ..\\data\\screening_design.csv
    -> writes ..\\data\\screening_results.csv

Measurement plan (from the Step 2 Gauge R&R):
  * Thickness / uniformity / RI: 49-point ellipsometer map, one reading.
  * Stress: average of 3 readings, because the stress gauge was marginal.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from pecvd import Recipe, Chamber
from metrology import Ellipsometer, StressGauge
from config import DEP_TIME_S

SEED = 33
CHAMBER = "T1-A"
STRESS_REPEATS = 3

# JMP column name -> simulator factor (matched case-insensitively)
FACTOR_COLUMNS = {
    "rf_power": "rf_power", "power": "rf_power",
    "pressure": "pressure",
    "temp": "temp", "temperature": "temp",
    "nh3_sih4": "nh3_sih4", "nh3/sih4": "nh3_sih4", "gas_ratio": "nh3_sih4",
    "spacing": "spacing",
}
RESPONSES = ["Dep_Rate", "WIW_NU", "RI", "Stress"]


def map_factor_columns(df):
    found = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in FACTOR_COLUMNS:
            found[FACTOR_COLUMNS[key]] = col
    missing = {"rf_power", "pressure", "temp", "nh3_sih4", "spacing"} - set(found)
    if missing:
        raise SystemExit(f"Could not find factor column(s) {sorted(missing)} in {list(df.columns)}.\n"
                         "Name the JMP factors RF_Power, Pressure, Temp, NH3_SiH4, Spacing.")
    return found


def run_design(df, seed=SEED):
    cols = map_factor_columns(df)
    rng = np.random.default_rng(seed)
    chamber = Chamber(CHAMBER, rng)
    chamber.clean()
    ellip, sg = Ellipsometer(rng), StressGauge(rng)

    out = df.copy()
    for r in RESPONSES:
        out[r] = np.nan
    for i, row in out.iterrows():
        recipe = Recipe(**{f: float(row[c]) for f, c in cols.items()})
        wafer = chamber.run(recipe)
        thk, ri = ellip.measure(wafer, "OP1")
        out.at[i, "Dep_Rate"] = round(thk.mean() / 10 / (DEP_TIME_S / 60), 2)     # nm/min
        out.at[i, "WIW_NU"] = round(thk.std(ddof=1) / thk.mean() * 100, 3)        # 1-sigma %
        out.at[i, "RI"] = round(float(ri.mean()), 4)
        out.at[i, "Stress"] = round(float(np.mean(
            [sg.measure(wafer, "OP1") for _ in range(STRESS_REPEATS)])), 1)       # MPa
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_doe.py <design.csv> [output.csv]")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(
        src.stem.replace("design", "results") + ("" if "design" in src.stem else "_results") + ".csv")
    df = pd.read_csv(src)
    out = run_design(df)
    out.to_csv(dst, index=False)
    print(out.to_string(index=False))
    print(f"\n{len(out)} runs processed on chamber {CHAMBER}. Saved: {dst}")


if __name__ == "__main__":
    main()
