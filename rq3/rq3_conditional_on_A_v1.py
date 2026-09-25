# -*- coding: utf-8 -*-
"""rq3_conditional_on_A_v1.py —— RQ3 主表唯一正典（2026-09-19 由 experiment_audit 移入 pipeline）。

口径（与论文 Figure 4 / RQ3 主表完全一致，已验证 658/750/638）：
在 931 个双链一致 base（prompt A 池 ∩ GT v1==GT v2）上，进一步只保留该模型
条件 A 生成 valid 且全签名匹配 GT_true v2 的 base，再测 B/C/D。即 controlled-for-fidelity。

字母映射：内部 cond B=加孔、C=干扰、D=去孔 → 论文 B=加孔、C=去孔、D=干扰。
输出与论文一致的前提是 labels 用现行正典文件（results_archive / experiments/runs/cadrille，
其中已含 2026-09-16 CC 同码噪声重标）。用旧 FINAL_FREEZE 副本会差 1 个样本（CC 去孔 hit 1.5% vs 1.1%）。

运行：python python rq3/rq3_conditional_on_A_v1.py
产出：本目录 rq3_conditional_on_A_v1.json
"""
import json, io, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_topobench.paths import LABELS, DATA   # noqa: E402

GT2 = LABELS / "gt_true_labels_v2_official.jsonl"
GT1 = LABELS / "test_topology_v1.json"
PROMPTS = DATA / "prompts" / "t4c_prompts_v2_expert.jsonl"
SETS = {
    "Text2CAD": (LABELS / "rq34_labels_rq3v2_v1.jsonl", "short"),
    "CAD-Coder": (LABELS / "cadcoder_labels_rq3v2_v1.jsonl", "uid"),
    "Cadrille-SFT": (LABELS / "labels_rq3v2_t4c_v1.jsonl", "uid"),
}

gt1 = {r["data_id"]: r["betti"] for r in json.loads(GT1.read_text(encoding="utf-8"))
       if r["success"]}
gt2 = {}
for l in io.open(GT2, encoding="utf-8"):
    r = json.loads(l)
    if r["ok"]:
        gt2[r["uid"]] = r["betti"]
consistent = {u for u in gt1 if u in gt2 and gt1[u] == gt2[u]}

prompts = {}
for l in io.open(PROMPTS, encoding="utf-8"):
    r = json.loads(l)
    prompts[(r["uid"].split("/")[1], r["condition"])] = r
bases = {u for (u, c) in prompts if c == "A"} & consistent


def boot_ci(vals, n=1000, seed=20260812):
    if len(vals) < 10:
        return None
    rng = random.Random(seed)
    ests = sorted(sum(rng.choice(vals) for _ in range(len(vals))) / len(vals)
                  for _ in range(n))
    return [round(ests[24], 4), round(ests[974], 4)]


def bucketize(r, gb, cond):
    tgt = [gb[0], gb[1] + 1, gb[2]] if cond == "B" else [gb[0], 0, gb[2]]
    if r["betti"] == tgt:
        return "hit"
    if cond == "B":
        if r["h1"] > gb[1]:
            return "dir_only"
        if r["h1"] == gb[1]:
            return "ignore"
        return "wrong_dir"
    if r["h1"] == 0:
        return "dir_only"
    if r["h1"] == gb[1]:
        return "ignore"
    if r["h1"] < gb[1]:
        return "partial"
    return "wrong_dir"


def main():
    out = {}
    for m, (fn, key) in SETS.items():
        labels = {}
        for l in io.open(fn, encoding="utf-8"):
            r = json.loads(l)
            labels[(r[key], r["cond"])] = r

        # 该模型 A 正确的 base
        a_ok = {u for u in bases
                if (u, "A") in labels and labels[(u, "A")]["ok"]
                and labels[(u, "A")]["betti"] == gt2[u]}
        rec = {"n_bases_consistent": len(bases), "n_bases_A_correct": len(a_ok)}

        for cond, tracks in [("B", [("B", None)]),
                             ("D", [("C_merged", None), ("C_clean", "clean"),
                                    ("C_conflict", "conflict")])]:
            for tname, et in tracks:
                rows, n_att = [], 0
                for u in sorted(a_ok):
                    r = labels.get((u, cond))
                    p = prompts.get((u, cond))
                    if r is None or not p:
                        continue
                    if et and p.get("edit_type") != et:
                        continue
                    n_att += 1
                    if not r["ok"] or not r["betti"]:
                        continue
                    rows.append(bucketize(r, gt2[u], cond))
                n = len(rows)
                d = {"attempted": n_att, "n_valid": n}
                for b in ["hit", "dir_only", "ignore", "partial", "wrong_dir"]:
                    xs = [1 if x == b else 0 for x in rows]
                    d[b] = round(sum(xs) / n, 4) if n else None
                    d[b + "_ci"] = boot_ci(xs)
                rec[tname] = d

        # Distractor：A 正确子集上的稳定性
        stable = []
        for u in sorted(a_ok):
            r, ra = labels.get((u, "C")), labels.get((u, "A"))
            if r and ra and r["ok"] and ra["ok"] and r["betti"] and ra["betti"]:
                stable.append(1 if r["betti"] == ra["betti"] else 0)
        rec["D_distractor"] = {"n_paired_valid": len(stable),
                               "sig_stable_vs_A": round(sum(stable) / len(stable), 4),
                               "sig_stable_ci": boot_ci(stable)}
        out[m] = rec

    json.dump(out, open(Path(__file__).resolve().parent /
                         "rq3_conditional_on_A_v1.json", "w"), indent=1)

    for m in SETS:
        r = out[m]
        print(f"\n===== {m} =====  A正确 base: {r['n_bases_A_correct']}/{r['n_bases_consistent']}")
        for tr in ["B", "C_merged", "C_clean", "C_conflict", "D_distractor"]:
            d = r[tr]
            if tr == "D_distractor":
                print(f"  {tr:10s} stable={d['sig_stable_vs_A']*100:.1f} (n={d['n_paired_valid']})")
            else:
                print(f"  {tr:10s} n={d['n_valid']:>3} | hit {d['hit']*100:5.1f}  dir {d['dir_only']*100:4.1f}  "
                      f"ignore {d['ignore']*100:5.1f}  partial {d['partial']*100:5.1f}  wrong {d['wrong_dir']*100:4.1f}")


if __name__ == "__main__":
    main()
