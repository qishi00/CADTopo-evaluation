# -*- coding: utf-8 -*-
"""build_model_specific_coverage.py —— 生成五模型各自的 covered/novel 划分。

每个模型一份 coverage/<model>.json：
  corpus_desc   : 语料说明（论文附录用）
  n_corpus_seqs : 训练序列数
  n_corpus_fps  : 去重后结构指纹数（v3：命令序列+EXT布尔操作）
  covered/novel : 测试集 8052 中该模型"见过/没见过"的 uid 清单

语料（全部核实，见 RESULTS_MASTER RQ2 附）：
  deepcad   : DC train 161,240（= 指纹缓存本身）
  text2cad  : T2C 官方 train 159,049（严格 ⊂ DC train，零泄漏）
  cadcoder  : 同 T2C（GRPO 用 T2C train 派生）
  cadrille  : 同 T2C（其 DeepCAD 分量；CAD-Recode 1M 分量在指纹词表外）
  skexgen   : 官方管线精确 survivors union 116,936
             （含 22,728 个 split 外 uid，其中仅 189 个有 vec 可打指纹）

运行：python build_model_specific_coverage.py
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
    """v3 指纹（与 frozen/fingerprint.py 即 coverage_flags_unified_v3.py 逐行一致）。"""
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
    split = json.load(open(ROOT / "data/data/train_val_test_split.json", encoding="utf-8"))
    dc_train = list(split["train"])
    t2c_train = sorted(json.load(open(ROOT / "text2cad_data/train_test_val.json",
                                      encoding="utf-8"))["train"])
    skx = json.load(open(RES / "skexgen_train_survivors.json", encoding="utf-8"))
    skx_union = sorted(set(skx["union"]))

    # SkexGen split 外 uid：用 vec 现算指纹（仅 189 个有 vec）
    extras = sorted(set(skx_union) - set(dc_train))
    fp_extra = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for tid, fpk in ex.map(fp_of, extras, chunksize=256):
            if fpk is not None:
                fp_extra[tid] = fpk

    corpora = {
        "deepcad":  {"desc": "DeepCAD train 161,240（官方 split）",
                     "uids": dc_train},
        "text2cad": {"desc": "Text2CAD 官方 train 159,049（⊂ DC train，零泄漏）",
                     "uids": t2c_train},
        "cadcoder": {"desc": "CAD-Coder：T2C train 派生（GRPO），同 text2cad",
                     "uids": t2c_train},
        "cadrille": {"desc": "Cadrille-SFT 的 DeepCAD 分量 = T2C 版；"
                             "CAD-Recode 1M 分量在指纹词表外（附录脚注）",
                     "uids": t2c_train},
        "skexgen":  {"desc": "SkexGen 官方管线 survivors（union 116,936；"
                             "split 外 uid 仅 vec 可指纹化的 189 个计入）",
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
        print(f"{model:10s} 语料序列={len(c['uids']):>7,} 指纹={len(fps):>6,} "
              f"covered={len(covered)} novel={len(novel)}")

    print("saved ->", OUT)


if __name__ == "__main__":
    main()
