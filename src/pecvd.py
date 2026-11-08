"""
Simulated single-wafer PECVD silicon nitride chamber.

For a recipe (5 factors) and a chamber, it returns the TRUE film on the wafer:
a 49-point thickness and RI map plus wafer stress. Measurement error is added
separately in metrology.py, so Gauge R&R can separate process and gauge noise.

Response directions follow public literature (see README):
  * Deposition rate rises with RF power and saturates; peaks vs pressure;
    falls as NH3/SiH4 rises (less silane); falls slightly with wider spacing
    and higher temperature.
  * Refractive index falls as NH3/SiH4 rises (N-rich film) and as RF power rises.
  * Compressive stress eases (moves toward tensile) with higher pressure,
    RF power and spacing; becomes more compressive at higher temperature.
  * Within-wafer uniformity improves with higher temperature and pressure and
    worsens with larger spacing and higher RF power. The radial profile can flip
    from center-thick to edge-thick, so uniformity has an optimum inside the
    factor space (curvature for the response surface study).
"""
from dataclasses import dataclass, field
import numpy as np

from config import (FACTORS, NOMINAL, DEP_TIME_S, WAFER_RADIUS_MM,
                    RING_RADII_MM, RING_COUNTS, CHAMBERS, NOISE, DRIFT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def coded(name, value):
    """Convert a factor value to coded units (-1 .. +1). Gas ratio is log-scaled."""
    lo, c, hi, _ = FACTORS[name]
    if name == "nh3_sih4":
        return np.log(value / c) / np.log(hi / c)
    return (value - c) / ((hi - lo) / 2.0)


def site_coordinates():
    """49-point polar pattern -> (x_mm, y_mm) arrays."""
    xs, ys = [], []
    for r, n in zip(RING_RADII_MM, RING_COUNTS):
        for k in range(n):
            theta = 2 * np.pi * k / n
            xs.append(r * np.cos(theta))
            ys.append(r * np.sin(theta))
    return np.array(xs), np.array(ys)


SITE_X, SITE_Y = site_coordinates()
RHO = np.hypot(SITE_X, SITE_Y) / WAFER_RADIUS_MM   # normalized radius 0..~1


@dataclass
class Recipe:
    rf_power: float = FACTORS["rf_power"][1]
    pressure: float = FACTORS["pressure"][1]
    temp: float = FACTORS["temp"][1]
    nh3_sih4: float = FACTORS["nh3_sih4"][1]
    spacing: float = FACTORS["spacing"][1]

    def coded(self):
        return {k: coded(k, getattr(self, k)) for k in FACTORS}


@dataclass
class WaferResult:
    chamber: str
    recipe: Recipe
    thickness_A: np.ndarray          # 49 sites, true thickness (Angstrom)
    ri: np.ndarray                   # 49 sites, true refractive index
    stress_mpa: float                # true wafer stress (negative = compressive)
    wafer_id: str = ""
    dep_time_s: float = DEP_TIME_S   # deposition time used for this wafer

    @property
    def mean_thk_A(self):
        return float(self.thickness_A.mean())

    @property
    def dep_rate_nm_min(self):
        return self.mean_thk_A / 10.0 / (self.dep_time_s / 60.0)

    @property
    def wiw_nu_pct(self):
        """Within-wafer non-uniformity, 1-sigma / mean (%)."""
        return float(self.thickness_A.std(ddof=1) / self.thickness_A.mean() * 100)

    @property
    def mean_ri(self):
        return float(self.ri.mean())

    def summary(self):
        d = {"wafer_id": self.wafer_id, "chamber": self.chamber}
        d.update({k: getattr(self.recipe, k) for k in FACTORS})
        d.update({
            "mean_thk_A": round(self.mean_thk_A, 1),
            "dep_rate_nm_min": round(self.dep_rate_nm_min, 2),
            "wiw_nu_pct": round(self.wiw_nu_pct, 3),
            "ri": round(self.mean_ri, 4),
            "stress_mpa": round(self.stress_mpa, 1),
        })
        return d


# ---------------------------------------------------------------------------
# Physics-informed response models (noise-free, chamber-neutral)
# ---------------------------------------------------------------------------
def mean_dep_rate(r: Recipe):
    """Mean deposition rate in nm/min."""
    p0 = 350.0                                            # power saturation scale (W)
    f_power = (1 - np.exp(-r.rf_power / p0)) / (1 - np.exp(-FACTORS["rf_power"][1] / p0))
    p_peak, width = 4.2, 3.0                              # rate peaks vs pressure
    f_press = np.exp(-((r.pressure - p_peak) / width) ** 2) / \
              np.exp(-((FACTORS["pressure"][1] - p_peak) / width) ** 2)
    s = 1.0 / (1.0 + r.nh3_sih4)                          # silane fraction
    s0 = 1.0 / (1.0 + FACTORS["nh3_sih4"][1])
    f_ratio = (s / s0) ** 0.55
    f_space = (FACTORS["spacing"][1] / r.spacing) ** 0.30
    f_temp = np.exp(-0.0012 * (r.temp - FACTORS["temp"][1]))
    return NOMINAL["dep_rate_nm_min"] * f_power * f_press * f_ratio * f_space * f_temp


def mean_ri(r: Recipe):
    x = r.coded()
    return (NOMINAL["ri"]
            - 0.12 * np.log(r.nh3_sih4 / FACTORS["nh3_sih4"][1])
            - 0.035 * x["rf_power"]
            + 0.010 * x["temp"]
            - 0.008 * x["pressure"]
            + 0.012 * x["rf_power"] * x["nh3_sih4"])


def stress(r: Recipe):
    x = r.coded()
    return (NOMINAL["stress_mpa"]
            + 60.0 * x["pressure"]
            + 50.0 * x["rf_power"]
            + 40.0 * x["spacing"]
            - 40.0 * x["temp"]
            + 30.0 * x["nh3_sih4"]
            + 15.0 * x["rf_power"] * x["pressure"]
            - 20.0 * x["rf_power"] ** 2)


def bowl_coefficient(r: Recipe):
    """Radial profile shape. >0 edge-thick, <0 center-thick, 0 = flat."""
    x = r.coded()
    return (0.045
            + 0.030 * x["spacing"]
            + 0.022 * x["rf_power"]
            - 0.025 * x["pressure"]
            - 0.012 * x["temp"]
            + 0.008 * x["spacing"] * x["pressure"])


EDGE_ROLLOFF = -0.008   # thinning in the outer ~5% of the wafer


# ---------------------------------------------------------------------------
# Chamber
# ---------------------------------------------------------------------------
class Chamber:
    def __init__(self, name, rng=None):
        self.name = name
        self.p = CHAMBERS[name]
        self.rng = rng if rng is not None else np.random.default_rng()
        self.wafers_since_clean = 0
        self._count = 0
        self.day = {"rate": 0.0, "ri": 0.0, "stress": 0.0}   # day-to-day shift

    def new_day(self):
        """Start a new production day: chamber clean plus a small random shift
        (e.g. after preventive maintenance or a new gas bottle)."""
        self.clean()
        self.day = {"rate": self.rng.normal(0, NOISE["rate_day_frac"]),
                    "ri": self.rng.normal(0, NOISE["ri_day"]),
                    "stress": self.rng.normal(0, NOISE["stress_day"])}

    def clean(self):
        self.wafers_since_clean = 0

    def run(self, recipe: Recipe, noise=True, dep_time_s=None) -> WaferResult:
        dep_time_s = DEP_TIME_S if dep_time_s is None else dep_time_s
        rng = self.rng
        n = self.wafers_since_clean

        # Mean thickness (A) with chamber offset and drift
        rate = (mean_dep_rate(recipe) * self.p["rate_mult"] * (1 + DRIFT["rate_per_wafer"] * n)
                * (1 + self.day["rate"]))
        if noise:
            rate *= 1 + rng.normal(0, NOISE["rate_wafer_frac"])
        mean_thk = rate * 10.0 * dep_time_s / 60.0

        # Radial profile + chamber tilt + edge roll-off
        b = bowl_coefficient(recipe) + self.p["bowl_off"]
        shape = (b * (RHO ** 2 - np.mean(RHO ** 2))
                 + self.p["tilt_x"] * SITE_X / WAFER_RADIUS_MM
                 + self.p["tilt_y"] * SITE_Y / WAFER_RADIUS_MM
                 + EDGE_ROLLOFF * np.clip((RHO - 0.93) / 0.05, 0, None) ** 2)
        thk = mean_thk * (1 + shape)
        if noise:
            thk *= 1 + rng.normal(0, NOISE["thk_site_frac"], size=thk.size)

        # Refractive index map (slightly higher at edge where film is denser)
        ri0 = mean_ri(recipe) + self.p["ri_off"] + DRIFT["ri_per_wafer"] * n + self.day["ri"]
        ri_map = ri0 + 0.004 * (RHO ** 2 - np.mean(RHO ** 2))
        if noise:
            ri_map = ri_map + rng.normal(0, NOISE["ri_wafer"]) \
                     + rng.normal(0, NOISE["ri_site"], size=ri_map.size)

        s = stress(recipe) + self.p["stress_off"] + self.day["stress"]
        if noise:
            s += rng.normal(0, NOISE["stress_wafer"])

        self.wafers_since_clean += 1
        self._count += 1
        return WaferResult(self.name, recipe, thk, ri_map, float(s),
                           wafer_id=f"{self.name}-W{self._count:04d}", dep_time_s=dep_time_s)


def make_fleet(seed=None):
    """All four chambers sharing one random generator."""
    rng = np.random.default_rng(seed)
    return {name: Chamber(name, rng) for name in CHAMBERS}


if __name__ == "__main__":
    fleet = make_fleet(seed=1)
    for name, ch in fleet.items():
        print(ch.run(Recipe()).summary())
