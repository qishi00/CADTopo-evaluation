# -*- coding: utf-8 -*-
"""aggregate_rq3n.py — RQ3-N five-class aggregation with sanity asserts.

RQ3-N is the paper's primary intervention protocol (Sec. 4.3): 600 bases from
the full test set on which all three text-conditioned models reproduce the
reference topology exactly (add-hole (1,0,0): 300, remove-hole (1,1,0): 300),
decoded greedily.  This script recomputes the five-class counts
(hit / partial / ignore / wrong_dir / other / invalid) from the shipped
per-sample labels and asserts equality with the frozen result.

用法:  python rq3/aggregate_rq3n.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_topobench.paths import LABELS, DATA   # noqa: E402

MODELS = ("text2cad", "cadcoder", "cadrille")
FROZEN = DATA / "results" / "rq3n_five_class_v1.json"


def classify(rec):
    gt, cond = rec["gt_betti"], rec["cond"]
    if not rec["ok"] or rec["betti"] is None:
        return "invalid"
    b = rec["betti"]
    target = [1, 1, 0] if cond == "B" else [1, 0, 0]
    if b == target:
        return "hit"
    if b == gt:
        return "ignore"
    if cond == "B" and b[1] >= 1:
        return "partial"
    if cond == "D" and b[1] == 0:
        return "partial"
    if cond == "B" and b[1] == 0:
        return "wrong_dir"
    if cond == "D" and b[1] >= 2:
        return "wrong_dir"
    return "other"


def main():
    out = {}
    for m in MODELS:
        f = LABELS / "rq3n" / f"{m}_labels_v1.jsonl"
        recs = [json.loads(l) for l in f.open(encoding="utf-8")]
        out[m] = {}
        for cond in ("B", "D"):
            sub = [r for r in recs if r["cond"] == cond]
            c = Counter(classify(r) for r in sub)
            n = len(sub)
            out[m][cond] = {"n": n, **{k: c.get(k, 0) for k in
                          ("hit", "partial", "ignore", "wrong_dir",
                           "other", "invalid")}}
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert out == frozen, (
        "RQ3-N mismatch:\n" + json.dumps({"recomputed": out, "frozen": frozen},
                                         indent=1))
    print("sanity 通过: RQ3-N five-class counts exactly match the frozen result")
    for m in MODELS:
        for cond in ("B", "D"):
            r = out[m][cond]
            print(f"  {m:<10} {cond}  n={r['n']:<4} hit={r['hit']}"
                  f" ignore={r['ignore']} invalid={r['invalid']}")


if __name__ == "__main__":
    main()
