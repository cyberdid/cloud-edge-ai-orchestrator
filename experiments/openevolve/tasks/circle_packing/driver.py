"""Sandbox driver for circle packing: call the candidate's ``run_packing()``."""

DRIVER = """
import json
import numpy as np
import candidate

centers, radii, reported = candidate.run_packing()
json.dump({
    "centers": np.asarray(centers, dtype=float).tolist(),
    "radii": np.asarray(radii, dtype=float).tolist(),
    "sum_radii": float(reported),
}, open("/output/result.json", "w"))
"""
