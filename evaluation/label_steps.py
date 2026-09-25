# -*- coding: utf-8 -*-
"""label_steps.py — 给一个目录下的所有 .step 文件提取 Betti 签名（cad_topobench 打标入口）

用法:
  python evaluation/label_steps.py <step_dir> [--out labels.jsonl] [--budget-sec 600]

输入: <step_dir> 下所有 *.step（递归）
输出: 每行 {"file", "ok", "betti", "error", "volume", "n_tris"}
      ok=False 表示 STEP 无法导入或不水密（对应论文中的 invalid / F0 类失败）
断点续跑: 已写入的 file 自动跳过。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cadquery as cq  # noqa: E402
from cad_topobench.reference import solid_betti, REFERENCE_VERSION  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget-sec", type=float, default=600.0)
    args = ap.parse_args()

    step_dir = Path(args.step_dir)
    out_path = Path(args.out) if args.out else step_dir / "labels_betti.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            done.add(json.loads(line)["file"])

    files = sorted(step_dir.rglob("*.step"))
    todo = [f for f in files if str(f) not in done]
    print(f"total={len(files)} done={len(done)} todo={len(todo)} "
          f"ref={REFERENCE_VERSION}", flush=True)

    t_end = time.time() + args.budget_sec
    out = out_path.open("a", encoding="utf-8")
    n = 0
    for f in todo:
        if time.time() > t_end:
            print("time budget reached", flush=True)
            break
        rec = {"file": str(f), "ok": False, "betti": None, "error": None,
               "volume": None, "n_tris": 0}
        try:
            shape = cq.importers.importStep(str(f)).val()
            r = solid_betti(shape)
            rec.update(ok=bool(r["ok"]), betti=r["betti"], error=r["error"],
                       volume=r["volume"], n_tris=r["n_tris"])
        except Exception as e:
            rec["error"] = f"step_import:{type(e).__name__}"
        out.write(json.dumps(rec) + "\n")
        out.flush()
        n += 1
        if n % 50 == 0:
            print(f"  +{n} ok={rec['ok']} betti={rec['betti']}", flush=True)
    out.close()
    print(f"labeled +{n} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
