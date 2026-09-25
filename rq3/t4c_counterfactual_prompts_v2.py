# -*- coding: utf-8 -*-
"""t4c_counterfactual_prompts_v2.py —— RQ3 反事实提示集扩产版（2026-08-24）

相对 v1 的变更：
- 候选池从分层 954 子集扩大到全集测试集（8052，GT 成功 7731：
  无孔 5245 / 带孔 2486）；
- N_HOLED/N_NOHOLED: 50 -> 550，v1 的 100 个 base 强制包含（超集），
  其余用新 seed 从剩余候选补齐；
- 只输出 expert 层（正文口径；四层 pooled 继续用 v1 附录数据）；
- pick() 由内置 hash() 改为 md5 稳定散列（v1 的内置 hash 随进程随机，
  v2 句子选择与 v1 可能不同 -> 重叠 uid 也需要重新生成，不使用 v1 旧输出）；
- 输出字段与 v1 完全一致（uid, level, group, condition, prompt, gt_betti,
  topo_status, edit_type），下游执行/标注/分析脚本零改动复用。

输出：<repo>/data/prompts/t4c_prompts_v2_expert.jsonl
"""
import csv, json, random, re, os, sys, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_topobench.paths import EXTERNAL, DATA   # noqa: E402

FILL_SEED = 20260824
N_HOLED, N_NOHOLED = 550, 550
CSV = str(EXTERNAL / "text2cad_v1.0.csv")
GT = str(DATA / "cad_topobench_v1.1.jsonl")
V1 = str(DATA / "prompts" / "t4c_prompts_v1.jsonl")
OUT = str(DATA / "prompts" / "t4c_prompts_v2_expert.jsonl")

# ---------- 干预句池（与 v1 逐字一致） ----------
B_INSTR = [
    " Finally, drill a through hole through the center of the part.",
    " As a last step, create a cylindrical through hole in the center of the part.",
    " Finally, cut a through hole passing through the center of the part.",
]
D_INSTR = [
    " Do not create any holes; keep the final part completely solid.",
    " Make sure the final part has no holes or openings of any kind.",
    " Skip any hole-cutting steps; the final part must be fully solid.",
]
C_INSTR = [
    " Finally, apply a smooth polished finish to all outer surfaces.",
    " Round the outer edges slightly for a better finish.",
    " Give the outer surfaces a matte texture.",
]

# ---------- 数据 ----------
gt = {}
for l in open(GT, encoding="utf-8"):
    d = json.loads(l)
    if d["gt"]["success"]:
        gt[d["id"]] = d["gt"]["betti"]

# 全集测试集候选池：GT jsonl 的 8052 个 id 即全集测试 uid8（成功标注 7731）
# v1 的 100 个 base（来自旧分层 954 子集）强制包含，其余用 FILL_SEED 补齐。
v1 = [json.loads(l) for l in open(V1, encoding="utf-8")]
v1_group = {}
for r in v1:
    v1_group[r["uid"]] = r["group"]
v1_holed = sorted(u for u, g in v1_group.items() if g == "holed")
v1_noholed = sorted(u for u, g in v1_group.items() if g == "noholed")

prompts = {}
with open(CSV, encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        short = row["uid"].split("/")[1]
        if short in gt:
            prompts[row["uid"]] = (row["expert"] or "").strip()

rng = random.Random(FILL_SEED)
cand = []
for uid in prompts:
    short = uid.split("/")[1]
    b = gt[short]
    cand.append((uid, b[1] > 0))
holed = sorted(u for u, h in cand if h)
noholed = sorted(u for u, h in cand if not h)
rng.shuffle(holed)
rng.shuffle(noholed)

# v1 base 优先，剩余从打乱池补齐（跳过已在 v1 的，避免重复）
def fill(v1_sel, pool, n):
    sel = list(v1_sel)
    for u in pool:
        if len(sel) >= n:
            break
        if u not in v1_group:
            sel.append(u)
    return sel

sel_h = fill(v1_holed, holed, N_HOLED)
sel_n = fill(v1_noholed, noholed, N_NOHOLED)
print(f"候选: holed={len(holed)} noholed={len(noholed)} -> 选取 {len(sel_h)}+{len(sel_n)}")

# 超集校验：v1 的 50+50 必须全部在内
v1_uids = {r["uid"] for r in v1}
v2_uids = set(sel_h) | set(sel_n)
missing = v1_uids - v2_uids
print(f"v1 超集校验: v1 uid={len(v1_uids)}, 不在 v2 的={len(missing)}")
assert not missing, f"超集校验失败: {list(missing)[:5]}"


def pick(pool, uid, lv):
    h = hashlib.md5(f"{uid}|{lv}|{pool[0][:7]}".encode()).hexdigest()
    return pool[int(h, 16) % len(pool)]


HOLE_EXPLICIT = re.compile(r"\b(holes?|through[- ]?holes?|bores?|openings?|perforations?|slots?)\b", re.I)
TOPO_IMPLIED = re.compile(r"\b(rings?|tubes?|tubular|hollow|pipes?|sleeves?|washers?|shells?|"
                          r"inner\s+circles?|cutouts?|cut[- ]?outs?|annular|donut|torus)\b", re.I)

def clean_base(t):
    return t.rstrip().rstrip(".")

def join_clause(base, cl):
    if cl.startswith(","):
        return clean_base(base) + cl + "."
    b = base.rstrip()
    if not b.endswith("."):
        b += "."
    return b + cl

def topo_status(base):
    if HOLE_EXPLICIT.search(base):
        return "explicit"
    if TOPO_IMPLIED.search(base):
        return "implied"
    return "none"

n_b_filtered = 0
rows = []
for uid, group in [(u, "holed") for u in sel_h] + [(u, "noholed") for u in sel_n]:
    short = uid.split("/")[1]
    base = prompts[uid]
    if not base:
        continue
    st = topo_status(base)
    rows.append(dict(uid=uid, level="expert", group=group, condition="A",
                     prompt=base, gt_betti=gt[short], topo_status=st))
    if group == "noholed":
        if st != "none":
            n_b_filtered += 1
        else:
            rows.append(dict(uid=uid, level="expert", group=group, condition="B",
                             prompt=join_clause(base, pick(B_INSTR, uid, "expert")),
                             gt_betti=gt[short], topo_status=st, edit_type="clean"))
    else:
        et = "clean" if st == "none" else "conflict"
        rows.append(dict(uid=uid, level="expert", group=group, condition="D",
                         prompt=join_clause(base, pick(D_INSTR, uid, "expert")),
                         gt_betti=gt[short], topo_status=st, edit_type=et))
    rows.append(dict(uid=uid, level="expert", group=group, condition="C",
                     prompt=join_clause(base, pick(C_INSTR, uid, "expert")),
                     gt_betti=gt[short], topo_status=st, edit_type="distractor"))
print(f"B 组过滤（显式孔词+隐含拓扑词）: {n_b_filtered}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"total prompts: {len(rows)} -> {OUT}")
from collections import Counter
print(Counter((r["group"], r["condition"]) for r in rows).most_common())
print("D 组 clean/conflict:",
      Counter(r["edit_type"] for r in rows if r["condition"] == "D"))
