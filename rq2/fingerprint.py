# -*- coding: utf-8 -*-
"""fingerprint.py — unified structural-fingerprint coverage flags, v3.

Upgrade over v2 (command-type column only): EXT rows additionally carry the
boolean operation type (column 15; 0=NewBody 1=Join 2=Cut 3=Intersect).
Rationale: the boolean operation is a discrete structural attribute that
directly determines topology (the same circle extruded as a boss vs. cut as
a hole), not a geometric parameter.

  fingerprint = (n_commands, blake2b over the int64 sequence of per-row
                 [command type (col 0), EXT operation / -1])
  EOS (3) rows are stripped; all remaining parameters (coordinates, sizes,
  distances) are discarded.
  Training fingerprint library = full DeepCAD train split, 161,240 sequences
      (T2C train 159,049 is a strict subset: zero cross-contamination; the
      2,191 DC-only sequences enter no T2C split).
  Test flags = DeepCAD test split, 8,052 (T2C test 8,046 is a subset).

Data source: the 'vec' dataset of <external>/cad_vec/{shard}/{fid}.h5.

Output: data/coverage/coverage_flags_unified_v3.json
Cache:  data/coverage/_unified_fp_cache_v3.pkl (resumable; not shared with v2)

Run:  python rq2/fingerprint.py
"""
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_topobench.paths import EXTERNAL, DATA   # noqa: E402

SPLIT = str(EXTERNAL / "train_val_test_split.json")   # DeepCAD official split
CADVEC = str(EXTERNAL / "cad_vec")
CACHE = str(DATA / "coverage" / "_unified_fp_cache_v3.pkl")
OUT = str(DATA / "coverage" / "coverage_flags_unified_v3.json")
WORKERS = 8
EXT_IDX, EOS_IDX = 5, 3


def strip_eos(a):
    a = np.asarray(a, dtype=np.int64)
    while a.shape[0] > 0 and a[-1, 0] == EOS_IDX:
        a = a[:-1]
    return a


def keys_of(tid):
    try:
        with h5py.File(os.path.join(CADVEC, tid + ".h5"), "r") as f:
            a = strip_eos(f["vec"][:])
        full = (a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest())
        # structural columns: [command type, EXT boolean op / -1]
        sig = np.zeros((a.shape[0], 2), dtype=np.int64)
        sig[:, 0] = a[:, 0]
        is_ext = a[:, 0] == EXT_IDX
        sig[is_ext, 1] = a[is_ext, 15]
        sig[~is_ext, 1] = -1
        struct = (a.shape[0],
                  hashlib.blake2b(sig.tobytes(), digest_size=16).hexdigest())
        return tid, full, struct
    except Exception:
        return tid, None, None


def main():
    sp = json.load(open(SPLIT, encoding="utf-8"))
    train_ids, test_ids = sp["train"], sp["test"]

    state = {"done": 0, "full": set(), "struct": set(), "fail": 0}
    if os.path.exists(CACHE):
        state = pickle.load(open(CACHE, "rb"))
        print(f"resume: train done={state['done']}", flush=True)

    todo = train_ids[state["done"]:]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for tid, fk, sk in ex.map(keys_of, todo, chunksize=256):
            state["done"] += 1
            if fk is None:
                state["fail"] += 1
            else:
                state["full"].add(fk)
                state["struct"].add(sk)
            if state["done"] % 10000 == 0:
                pickle.dump(state, open(CACHE, "wb"))
                print(f"  train {state['done']}/{len(train_ids)} "
                      f"struct={len(state['struct'])}", flush=True)
    pickle.dump(state, open(CACHE, "wb"))
    print(f"train done: {state['done']} (failed {state['fail']}) "
          f"distinct struct={len(state['struct'])}", flush=True)

    flags, miss = {}, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for tid, fk, sk in ex.map(keys_of, test_ids, chunksize=256):
            if sk is None:
                flags[tid] = dict(gt_struct_in_train=None, gt_full_in_train=None)
                miss += 1
            else:
                flags[tid] = dict(gt_struct_in_train=sk in state["struct"],
                                  gt_full_in_train=fk in state["full"])

    n_cov = sum(1 for v in flags.values() if v["gt_struct_in_train"])
    n_ok = sum(1 for v in flags.values() if v["gt_struct_in_train"] is not None)
    json.dump(dict(
        note="unified v3: structural fingerprint = (command-type sequence + EXT boolean op, "
             "all geometric parameters dropped, EOS stripped); "
             "training library = DC train 161240 (T2C train is a subset)",
        n_train=state["done"], n_train_struct=len(state["struct"]),
        flags=flags),
        open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"covered {n_cov}/{n_ok} ({n_cov / max(1, n_ok) * 100:.1f}%) "
          f"miss={miss} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
