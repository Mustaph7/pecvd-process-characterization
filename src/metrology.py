"""
Simulated metrology: an ellipsometer (thickness + RI) and a stress gauge.

Each measurement = true value + operator bias + wafer x operator interaction
+ repeatability noise. This is the structure a crossed Gauge R&R study
separates into repeatability, reproducibility and part-to-part variation.
"""
import numpy as np
from config import METROLOGY


class Ellipsometer:
    def __init__(self, rng=None):
        self.p = METROLOGY["ellipsometer"]
        self.rng = rng if rng is not None else np.random.default_rng()
        self._interaction = {}     # (wafer_id, operator) -> fixed offset

    def _pxo(self, wafer_id, operator):
        key = (wafer_id, operator)
        if key not in self._interaction:
            self._interaction[key] = self.rng.normal(0, self.p["part_x_op_sd_A"])
        return self._interaction[key]

    def measure(self, wafer, operator="OP1"):
        """Return measured 49-point thickness (A) and RI arrays."""
        n = wafer.thickness_A.size
        thk = (wafer.thickness_A
               + self.p["operator_bias_A"][operator]
               + self._pxo(wafer.wafer_id, operator)
               + self.rng.normal(0, self.p["thk_repeat_sd_A"], size=n))
        ri = (wafer.ri
              + self.p["operator_ri_bias"][operator]
              + self.rng.normal(0, self.p["ri_repeat_sd"], size=n))
        return thk, ri


class StressGauge:
    def __init__(self, rng=None):
        self.p = METROLOGY["stress_gauge"]
        self.rng = rng if rng is not None else np.random.default_rng()

    def measure(self, wafer, operator="OP1"):
        return (wafer.stress_mpa
                + self.p["operator_bias_mpa"][operator]
                + self.rng.normal(0, self.p["repeat_sd_mpa"]))
