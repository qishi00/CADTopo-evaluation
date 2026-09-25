# -*- coding: utf-8 -*-
"""build_model_specific_coverage.py — build the per-model covered/novel split.

Writes one coverage/<model>.json per model:
  corpus_desc   : corpus description (for the paper appendix)
  n_corpus_seqs : number of training sequences
  n_corpus_fps  : number of distinct structural fingerprints (v3: command
                  sequence + EXT boolean op)
  covered/novel : uid lists of the 8,052 test samples the model has / has not
                  "seen" during training

Corpora (all verified):
  deepcad   : DC train 161,240 (= the fingerprint cache itself)
  text2cad  : T2C official train 159,049 (strict subset of DC train, zero leakage)
  cadcoder  : same as T2C (derived from the T2C train split for GRPO)
  cadrille  : same as T2C for its DeepCAD component; the CAD-Recode 1M
              component is outside the fingerprint vocabulary
  skexgen   : exact survivors union of the official pipeline, 116,936
              (includes 22,728 uids outside the official split, of which only
              189 have a vectorization usable for fingerprinting)

Run:  python rq2/build_model_specific_coverage.py
"""
import hashlib
import io
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_topobench.paths import EXTERNAL, DATA   # noqa: E402

RES = DATA
AUDIT = DATA / "audit"
OUT = Path(__file__).parent / "coverage"
EXT_IDX, EOS_IDX = 5, 3
CADVEC = str(EXTERNAL / "cad_vec")


def fp_of(tid):
    """v3 fingerprint (identical to rq2/fingerprint.py)."""
    try:
        with h5py.File(os.path.join(CADVEC, tid + ".h5"), "r") as f:
            a = np.asarray(f["vec"][:], dtype=np.int64)
        while a.shape[0] > 0 and a[-1, 0] == EOS_IDX:
            a = a[:-1]
        sig = np.zeros((a.shape[0], 2), dtype=np.int64)
        sig[:, 0] = a[:, 0]
        is_ext = a[:, 0] == EXT_IDX
        sig[is_ext, 1] = a[is_ext, 15]
        sig[~is_ext, 1] = -1
        return tid, (a.shape[0],
                     hashlib.blake2b(sig.tobytes(), digest_size=16).hexdigest())
    except Exception:
        return tid, None


def main():
    OUT.mkdir(exist_ok=True)
    st = pickle.load(open(AUDIT / "_p8_fp_cache.pkl", "rb"))
    train_fp = st["train_fp"]                      # 'shard/uid' -> (n_rows, hash)
    split = json.load(open(EXTERNAL / "train_val_test_split.json", encoding="utf-8"))
    dc_train = list(split["train"])
    t2c_train = sorted(json.load(open(EXTERNAL / "text2cad_train_split.json",
                                      encoding="utf-8"))["train"])
    skx = json.load(open(AUDIT / "skexgen_train_survivors.json", encoding="utf-8"))
    skx_union = sorted(set(skx["union"]))

    # SkexGen uids outside the official split: fingerprint on the fly
    # (only 189 of them have a vectorization)
    extras = sorted(set(skx_union) - set(dc_train))
    fp_extra = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for tid, fpk in ex.map(fp_of, extras, chunksize=256):
            if fpk is not None:
                fp_extra[tid] = fpk

    corpora = {
        "deepcad":  {"desc": "DeepCAD train 161,240 (official split)",
                     "uids": dc_train},
        "text2cad": {"desc": "Text2CAD official train 159,049 (subset of DC train, zero leakage)",
                     "uids": t2c_train},
        "cadcoder": {"desc": "CAD-Coder: derived from the T2C train split (GRPO), same as text2cad",
                     "uids": t2c_train},
        "cadrille": {"desc": "Cadrille-SFT DeepCAD component = T2C version; "
                             "the CAD-Recode 1M component is outside the "
                             "fingerprint vocabulary (appendix footnote)",
                     "uids": t2c_train},
        "skexgen":  {"desc": "SkexGen official-pipeline survivors (union 116,936; "
                             "of the out-of-split uids only the 189 with a "
                             "usable vectorization are counted)",
                     "uids": skx_union},
    }

    test_fp = {tid.split("/")[-1]: fp for tid, (fp, n) in st["test"].items()}

    for model, c in corpora.items():
        fps = {train_fp[u] for u in c["uids"] if u in train_fp}
        if model == "skexgen":
            fps |= set(fp_extra.values())
        covered, novel = [], []
        for uid, fp in sorted(test_fp.items()):
            if fp is None:
                continue
            (covered if fp in fps else novel).append(uid)
        out = {"model": model, "corpus_desc": c["desc"],
               "n_corpus_seqs": len(c["uids"]),
               "n_corpus_fps": len(fps),
               "n_covered": len(covered), "n_novel": len(novel),
               "covered": covered, "novel": novel}
        json.dump(out, io.open(OUT / f"{model}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"{model:10s} corpus_seqs={len(c['uids']):>7,} fingerprints={len(fps):>6,} "
              f"covered={len(covered)} novel={len(novel)}")

    print("saved ->", OUT)


if __name__ == "__main__":
    main()
