# -*- coding: utf-8 -*-
"""Central path configuration.

All data locations are resolved relative to CAD_TOPOBENCH_ROOT (environment
variable). Default: the repository root (this file's parent's parent).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "CAD_TOPOBENCH_ROOT",
    Path(__file__).resolve().parent.parent)).resolve()

# External public datasets (download instructions in README.md)
EXTERNAL = ROOT / "external"
DEEPCAD_ROOT = EXTERNAL / "DeepCAD"          # DeepCAD repo (cadlib + create_CAD)
CAD_JSON = EXTERNAL / "cad_json"             # DeepCAD json sequences
CAD_VEC = EXTERNAL / "cad_vec"               # DeepCAD 8-bit quantized vectors
TEXT2CAD_CSV = EXTERNAL / "text2cad_v1.0.csv"  # Text2CAD prompt table

# Derived data shipped with this repository
DATA = ROOT / "data"
LABELS = DATA / "labels"
COVERAGE = DATA / "coverage"
RESULTS = DATA / "results"
