# -*- coding: utf-8 -*-
"""RQ2 标准化差距（Table 7）：控制拓扑复杂度 × 命令长度后的 covered-novel 差距。

方法（与论文 §5.2 一致）：
  cell = C_topo层(3) × 命令长度tercile(3) 共 9 格；
  仅保留 covered 和 novel 两侧样本数都 >= MIN_CELL(=5) 的格；
  权重 = novel 组的 cell 分布（标准化到 novel 的复杂度分布上）；
  置信区间 = 样本级 bootstrap 1000 次（seed=20260828）。

每个模型使用自己的训练语料覆盖（coverage/<model>.json）。

用法：
    python standardize_gap.py
输出：
    code_final/RQ2/results/rq2_standardized_gap.json
内置 sanity 断言（不一致直接 AssertionError）：
  gap / n_cells / retained 必须与冻结结果【精确相等】；
  bootstrap CI 因随机序列复现路径不同，允许 ±0.01 容差。
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

# ---- 指纹/命令数缓存（P8 冻结产物：测试集 tid -> (fingerprint, ncmd)） ----
st = pickle.load(io.open(AUDIT / "_p8_fp_cache.pkl", "rb"))
ncmd = {tid.split("/")[-1]: n for tid, (fp, n) in st["test"].items()}
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
    """标准化 matched gap：novel 分布加权、MIN_CELL 过滤。"""
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

    # ---- sanity 断言 ----
    fz = FROZEN[model]
    for k in ("gap", "n_cells", "retained"):
        assert rq2b[k] == fz[k], f"{model} rq2b[{k}]: {rq2b[k]} != frozen {fz[k]}"
    for i in (0, 1):
        assert abs(rq2b["ci"][i] - fz["ci"][i]) <= CI_TOL, \
            f"{model} rq2b CI: {rq2b['ci']} vs frozen {fz['ci']} (tol {CI_TOL})"

    result["models"][model] = {"corpus_desc": covd["corpus_desc"], "rq2b": rq2b}
    print(f"[OK] {model:<12} sanity 通过 | T7 gap={rq2b['gap']} "
          f"CI={rq2b['ci']} (frozen {fz['ci']}) retained={rq2b['retained']} "
          f"cells={rq2b['n_cells']}")

json.dump(result, io.open(OUT / "rq2_standardized_gap.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nsaved ->", OUT / "rq2_standardized_gap.json")
print("全部模型 sanity 断言通过（gap/n_cells/retained 精确一致，CI 在容差内）。")
