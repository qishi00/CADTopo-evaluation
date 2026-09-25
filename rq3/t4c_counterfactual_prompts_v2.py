# -*- coding: utf-8 -*-
"""t4c_counterfactual_prompts_v2.py — RQ3 counterfactual prompt set, expanded (2026-08-24)

Changes relative to v1:
- Candidate pool enlarged from the stratified 954-subset to the full test set
  (8052 inputs, 7731 with valid GT: 5245 without through-holes / 2486 with holes);
- N_HOLED/N_NOHOLED: 50 -> 550; the 100 v1 bases are forcibly included (superset),
  the remainder filled from the shuffled pool with a new seed;
- Only the expert level is emitted (main-text protocol; the four-level pooled
  analysis continues to use the v1 appendix data);
- pick() switched from the built-in hash() to a stable md5 hash (v1's built-in
  hash is randomized per process, so v2 sentence selection may differ from v1 ->
  overlapping uids are also regenerated; v1 outputs are not reused);
- Output fields are exactly identical to v1 (uid, level, group, condition, prompt,
  gt_betti, topo_status, edit_type); downstream execution/labeling/analysis
  scripts are reused unchanged.

Output: <repo>/data/prompts/t4c_prompts_v2_expert.jsonl
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

# ---------- intervention-sentence pools (verbatim from v1) ----------
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

# ---------- data ----------
gt = {}
for l in open(GT, encoding="utf-8"):
    d = json.loads(l)
    if d["gt"]["success"]:
        gt[d["id"]] = d["gt"]["betti"]

# full-test candidate pool: the 8,052 ids of the GT jsonl are the full test
# uid8s (7,731 successfully labeled). The 100 v1 bases (from the old
# stratified 954 subset) are always included; the rest is filled with FILL_SEED.
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

# v1 bases first, remainder filled from the shuffled pool (skip uids already in v1)
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
print(f"candidates: holed={len(holed)} noholed={len(noholed)} -> selected {len(sel_h)}+{len(sel_n)}")

# superset check: all v1 50+50 bases must be included
v1_uids = {r["uid"] for r in v1}
v2_uids = set(sel_h) | set(sel_n)
missing = v1_uids - v2_uids
print(f"v1 superset check: v1 uid={len(v1_uids)}, missing from v2={len(missing)}")
assert not missing, f"superset check failed: {list(missing)[:5]}"


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
print(f"B-group filter (explicit hole words + implied-topology words): {n_b_filtered}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"total prompts: {len(rows)} -> {OUT}")
from collections import Counter
print(Counter((r["group"], r["condition"]) for r in rows).most_common())
print("D-group clean/conflict:",
      Counter(r["edit_type"] for r in rows if r["condition"] == "D"))
