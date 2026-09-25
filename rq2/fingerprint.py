# -*- coding: utf-8 -*-
"""coverage_flags_unified_v3.py —— 统一结构指纹覆盖标记 v3（2026-08-23）

在 v2（纯命令类型列）基础上升级：EXT 行附加布尔操作类型（col15，
0=NewBody 1=Join 2=Cut 3=Intersect）。理由：布尔操作是离散结构属性，
直接决定拓扑（同一个圆拉伸成凸台 vs 切成孔），不是几何参数。

  指纹 = (命令数, blake2b(每行 [col0命令类型, EXT操作/-1] 的 int64 序列))
  EOS(3) 行剥除，其余参数（坐标/尺寸/距离）全部丢弃。
  训练指纹库 = DeepCAD train split 全集 161,240（T2C train 159,049 为真子集，
               0 交叉污染；DC 独有 2191 未进入 T2C 任何 split）。
  测试标记 = DeepCAD test split 8052（T2C test 8046 为真子集）。

数据源：<external>/cad_vec/{shard}/{fid}.h5 的 'vec'。

输出：results/coverage_flags_unified_v3.json
缓存：results/_unified_fp_cache_v3.pkl（断点续跑，与 v2 缓存不共用）

运行：python rq2/fingerprint.py
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
        # 结构列: [命令类型, EXT布尔操作/-1]
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
        print(f"续跑: train done={state['done']}", flush=True)

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
    print(f"train 完成: {state['done']} (失败 {state['fail']}) "
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
        note="unified v3 2026-08-23: 结构指纹=(命令类型序列+EXT布尔操作, "
             "几何参数全丢, EOS剥除); 训练库=DC train 161240 (T2C train 真子集)",
        n_train=state["done"], n_train_struct=len(state["struct"]),
        flags=flags),
        open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"covered {n_cov}/{n_ok} ({n_cov / max(1, n_ok) * 100:.1f}%) "
          f"miss={miss} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
