# -*- coding: utf-8 -*-
"""RQ2 standardized gap (Figure 3(c)): the covered-novel fidelity gap after
controlling for topological complexity x command-sequence length.

Method (identical to Sec. 5.2 of the paper):
  cell = 3 C_topo layers x 3 command-length terciles = 9 cells;
  only cells with >= MIN_CELL(=5) samples on BOTH the covered and the novel
  side are kept;
  weights = the novel group's cell distribution (the covered side is
  standardized onto the novel complexity distribution);
  confidence intervals = sample-level bootstrap, 1000 resamples
  (seed=20260828).

Each model uses its own training-corpus coverage (coverage/<model>.json).

Usage:
    python rq2/standardize_gap.py
Output:
    rq2/results/rq2_standardized_gap.json
Built-in sanity asserts (viations raise AssertionError):
  gap / n_cells / retained must match the frozen result EXACTLY;
  bootstrap CIs may deviate by up to 0.01 across replay paths.
"""
import io
import json
import pickle
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_topobench.paths import LABELS, DATA   # noqa: E402

RES = LABELS
AUDIT = DATA / "audit"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

MIN_CELL, N_BOOT, SEED = 5, 1000, 20260828
CI_TOL = 0.01

MODELS = {
    "DeepCAD": ("deepcad.json", "deepcad_gen_labels_v2_official.jsonl"),
    "SkexGen": ("skexgen.json", "skexgen_labels_v1.jsonl"),
    "Text2CAD": ("text2cad.json", "text2cad_labels_full_expert_v1.jsonl"),
    "CAD-Coder": ("cadcoder.json", "cadcoder_labels_full_v1.jsonl"),
    "Cadrille-SFT": ("cadrille.json", "cadrille_labels_full_v1.jsonl"),
}

FZ_BD = json.load(io.open(AUDIT / "rq2bd_ctopo_gttrue_v2.json", encoding="utf-8"))["models"]
P17_T2C = json.load(io.open(AUDIT / "p17_full_rq2_t2c.json", encoding="utf-8"))["models"]
P17_SKX = json.load(io.open(AUDIT / "p17_full_rq2_skexgen.json", encoding="utf-8"))
FROZEN = {
    "DeepCAD": FZ_BD["DeepCAD"]["rq2b"],
    "SkexGen": P17_SKX["skx_union"]["rq2b"],
    "Text2CAD": P17_T2C["Text2CAD"]["t2c_corpus"]["rq2b"],
    "CAD-Coder": P17_T2C["CAD-Coder"]["t2c_corpus"]["rq2b"],
    "Cadrille-SFT": P17_T2C["Cadrille-SFT"]["t2c_corpus"]["rq2b"],
}

# ---- fingerprint / command-count cache (frozen artifact: test tid -> [ncmd, fp hash]) ----
_fp = json.load(io.open(AUDIT / "test_fingerprint_ncmd_v1.json", encoding="utf-8"))
ncmd = {tid.split("/")[-1]: v[0] for tid, v in _fp.items()}
allnc = sorted(ncmd.values())
TERC = (float(np.percentile(allnc, 33.333)), float(np.percentile(allnc, 66.667)))


def ctopo_layer(b):
    c = (b[0] - 1) + b[1] + b[2]
    return "C0" if c == 0 else ("C12" if c <= 2 else "C3p")


def load_labels(path):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        r = json.loads(line)
        u = r.get("uid") or r.get("id")
        if not u:
            continue
        out[u[-8:]] = tuple(int(x) for x in r["betti"][:3]) \
            if r.get("ok") and isinstance(r.get("betti"), list) else None
    return out


def mgap(rows):
    """Standardized matched gap: novel-distribution weighting, MIN_CELL filter."""
    agg = {}
    for r in rows:
        t = 0 if r["ncmd"] <= TERC[0] else (1 if r["ncmd"] <= TERC[1] else 2)
        k = ((r["layer"], t), bool(r["covered"]))
        a = agg.setdefault(k, [0, 0])
        a[0] += r["hit"]
        a[1] += 1
    nov = {c: v for (c, cov_), v in agg.items() if not cov_}
    cells = [c for c in nov if nov[c][1] >= MIN_CELL
             and agg.get((c, True), [0, 0])[1] >= MIN_CELL]
    if not cells:
        return None
    mass = sum(nov[c][1] for c in cells)
    total = sum(v[1] for v in nov.values())
    mc = mn = 0.0
    for c in cells:
        w = nov[c][1] / mass
        mc += w * agg[(c, True)][0] / agg[(c, True)][1]
        mn += w * nov[c][0] / nov[c][1]
    return {"gap": mc - mn, "n_cells": len(cells), "retained": mass / total}


gtt = {}
for line in io.open(RES / "gt_true_labels_v2_official.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("ok") and isinstance(r.get("betti"), list):
        gtt[r["uid"][-8:]] = tuple(int(x) for x in r["betti"][:3])

result = {"tercile_ncmd": TERC, "min_cell": MIN_CELL, "n_boot": N_BOOT,
          "seed": SEED, "models": {}}
for model, (cov_file, lab_file) in MODELS.items():
    covd = json.load(io.open(DATA / "coverage" / cov_file, encoding="utf-8"))
    covered_set = set(covd["covered"])
    novel_set = set(covd["novel"])
    pred = load_labels(RES / lab_file)

    rows = []
    for uid, g in gtt.items():
        p = pred.get(uid)
        nc = ncmd.get(uid)
        if p is None or nc is None or (uid not in covered_set and uid not in novel_set):
            continue
        rows.append({"uid": uid, "hit": int(g == p), "covered": uid in covered_set,
                     "layer": ctopo_layer(g), "ncmd": nc})

    base = mgap(rows)
    uids = sorted({r["uid"] for r in rows})
    by = {r["uid"]: r for r in rows}
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        s = rng.choice(uids, size=len(uids), replace=True)
        b = mgap([by[u] for u in s])
        if b:
            boots.append(b["gap"])
    rq2b = {"gap": round(base["gap"], 4), "n_cells": base["n_cells"],
            "retained": round(base["retained"], 4),
            "ci": [round(float(np.percentile(boots, 2.5)), 4),
                   round(float(np.percentile(boots, 97.5)), 4)]}

    # ---- sanity asserts ----
    fz = FROZEN[model]
    for k in ("gap", "n_cells", "retained"):
        assert rq2b[k] == fz[k], f"{model} rq2b[{k}]: {rq2b[k]} != frozen {fz[k]}"
    for i in (0, 1):
        assert abs(rq2b["ci"][i] - fz["ci"][i]) <= CI_TOL, \
            f"{model} rq2b CI: {rq2b['ci']} vs frozen {fz['ci']} (tol {CI_TOL})"

    result["models"][model] = {"corpus_desc": covd["corpus_desc"], "rq2b": rq2b}
    print(f"[OK] {model:<12} sanity passed | Fig3c gap={rq2b['gap']} "
          f"CI={rq2b['ci']} (frozen {fz['ci']}) retained={rq2b['retained']} "
          f"cells={rq2b['n_cells']}")

json.dump(result, io.open(OUT / "rq2_standardized_gap.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nsaved ->", OUT / "rq2_standardized_gap.json")
print("All models passed sanity asserts (gap/n_cells/retained exact, CI within tolerance).")
