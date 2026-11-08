"""
Configuration for the simulated PECVD silicon nitride (SiNx) chamber.

All values are ILLUSTRATIVE. Factor ranges follow public literature on
single-wafer parallel-plate PECVD nitride (see README references). Model
coefficients were chosen so each response moves in the direction reported
in that literature; they are not specifications of any real tool.
"""

# ---------------------------------------------------------------------------
# Process factors: name -> (low, center, high, units)
# ---------------------------------------------------------------------------
FACTORS = {
    "rf_power":  (300.0, 450.0, 600.0, "W"),
    "pressure":  (2.0,   3.5,   5.0,   "Torr"),
    "temp":      (300.0, 350.0, 400.0, "C"),
    "nh3_sih4":  (2.0,   4.0,   8.0,   "ratio"),   # NH3/SiH4 flow ratio (log-scaled)
    "spacing":   (300.0, 450.0, 600.0, "mils"),    # showerhead-to-wafer gap
}

# Deposition time used for every run (seconds)
DEP_TIME_S = 330.0

# ---------------------------------------------------------------------------
# Nominal responses at the center point
# ---------------------------------------------------------------------------
NOMINAL = {
    "dep_rate_nm_min": 55.0,     # ~9 A/s
    "ri":              2.00,     # refractive index at 633 nm
    "stress_mpa":     -150.0,    # negative = compressive
}

# ---------------------------------------------------------------------------
# Wafer and measurement pattern
# ---------------------------------------------------------------------------
WAFER_RADIUS_MM = 150.0          # 300 mm wafer
EDGE_EXCLUSION_MM = 3.0
# Standard 49-point polar pattern: center + rings of 8, 16, 24 points
RING_RADII_MM = [0.0, 49.0, 98.0, 147.0]
RING_COUNTS = [1, 8, 16, 24]

# ---------------------------------------------------------------------------
# Chambers: two twin-chamber tools -> four chambers.
# Offsets make each chamber slightly different (the matching problem).
# ---------------------------------------------------------------------------
CHAMBERS = {
    "T1-A": {"rate_mult": 1.000, "ri_off":  0.000, "stress_off":   0.0, "bowl_off":  0.000, "tilt_x":  0.002, "tilt_y":  0.000},
    "T1-B": {"rate_mult": 1.012, "ri_off": -0.004, "stress_off":   8.0, "bowl_off":  0.004, "tilt_x": -0.001, "tilt_y":  0.002},
    "T2-A": {"rate_mult": 0.985, "ri_off":  0.006, "stress_off": -12.0, "bowl_off": -0.003, "tilt_x":  0.000, "tilt_y": -0.002},
    "T2-B": {"rate_mult": 1.021, "ri_off": -0.002, "stress_off":  15.0, "bowl_off":  0.007, "tilt_x":  0.003, "tilt_y":  0.001},
}

# ---------------------------------------------------------------------------
# Process noise (true film variation, before measurement)
# ---------------------------------------------------------------------------
NOISE = {
    "rate_wafer_frac": 0.004,    # wafer-to-wafer rate noise (fraction)
    "thk_site_frac":   0.0025,   # site-to-site random thickness noise (fraction)
    "ri_wafer":        0.002,
    "ri_site":         0.001,
    "stress_wafer":    5.0,      # MPa
}

# Chamber drift between cleans (wall buildup)
DRIFT = {
    "rate_per_wafer": -1.5e-5,   # fractional rate change per wafer since clean
    "ri_per_wafer":    2.0e-6,
}

# ---------------------------------------------------------------------------
# Metrology (used for Gauge R&R). Values in Angstrom / RI units / MPa.
# ---------------------------------------------------------------------------
METROLOGY = {
    "ellipsometer": {
        "thk_repeat_sd_A": 2.0,          # repeatability
        "ri_repeat_sd": 0.0008,
        "operator_bias_A": {"OP1": 0.0, "OP2": 2.5, "OP3": -1.5},   # reproducibility
        "operator_ri_bias": {"OP1": 0.0, "OP2": 0.0006, "OP3": -0.0004},
        "part_x_op_sd_A": 1.0,           # wafer x operator interaction
    },
    "stress_gauge": {
        "repeat_sd_mpa": 3.0,
        "operator_bias_mpa": {"OP1": 0.0, "OP2": 2.0, "OP3": -1.0},
    },
}
