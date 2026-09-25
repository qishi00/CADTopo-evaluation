# -*- coding: utf-8 -*-
"""aggregate_rq1.py — reproduce the paper's RQ1 headline numbers from the
shipped per-sample topology labels, and assert equality with the frozen
result (data/results/rq1_paper_numbers_v1.json).

Covers:
  Table 1   (main text)   invalid rate (F0), per-component Betti accuracy,
                          exact signature fidelity (valid), signature+invalid
                          fidelity — reference: GT_true (official builder)
  Figure 2(a)             signature fidelity by C_topo stratum (0 / 1-2 / >=3)
  Figure 2(b)             error-direction distribution (under / over / mixed,
                          Eq. 2 of the paper) over valid signature mismatches
  Figure 3(a)             covered vs novel signature fidelity (raw gap),
                          coverage = GT structural fingerprint present in the
                          DeepCAD training corpus (data/coverage/
                          gt_structure_flags_v1.json)
  Figure 3(b)             covered vs novel fidelity within each C_topo stratum
  Table 6   (appendix)    chi-fidelity (Euler characteristic match) on valid
                          generations, same labels as Table 1
  Table 7   (appendix)    behavior of chi on signature-error samples:
                          chi-invisible errors, multi-component errors, and
                          how many of the latter have delta-chi != 0

Usage:  python rq1/aggregate_rq1.py
Exits 0 and prints PASS when the recomputation matches the frozen result.
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cad_topobench.paths import LABELS, DATA, RESULTS  # noqa: E402

GT = LABELS / "gt_true_labels_v2_official.jsonl"
GEN = {
    "DeepCAD": LABELS / "deepcad_gen_labels_v2_official.jsonl",
    "SkexGen": LABELS / "skexgen_labels_v1.jsonl",
    "Text2CAD": LABELS / "text2cad_labels_full_expert_v1.jsonl",
    "CAD-Coder": LABELS / "cadcoder_labels_full_v1.jsonl",
    "Cadrille-SFT": LABELS / "cadrille_labels_full_v1.jsonl",
}
FLAGS = DATA / "coverage" / "gt_structure_flags_v1.json"
FROZEN = RESULTS / "rq1_paper_numbers_v1.json"

MODELS = ["DeepCAD", "SkexGen", "Text2CAD", "CAD-Coder", "Cadrille-SFT"]


def ctopo_layer(b):
    c = (b[0] - 1) + b[1] + b[2]
    return "C0" if c == 0 else ("C12" if c <= 2 else "C3p")


def load_gt():
    gt = {}
    for line in io.open(GT, encoding="utf-8"):
        d = json.loads(line)
        if d.get("ok") and isinstance(d.get("betti"), list):
            gt[d["uid"][-8:]] = tuple(int(x) for x in d["betti"][:3])
    return gt


def load_preds(path, model):
    preds = {}
    for line in io.open(path, encoding="utf-8"):
        r = json.loads(line)
        if model == "SkexGen" and not (r.get("cond") is None and r.get("k") is None):
            continue  # baseline generations only (the k=5 rows are a
        # different track and are not part of Table 1 / Figure 2)
        u = r.get("uid")
        if u is None:
            continue
        preds[u[-8:]] = (tuple(int(x) for x in r["betti"][:3])
                         if r.get("ok") and isinstance(r.get("betti"), list)
                         else None)
    return preds


def direction(gp):
    """Eq. 2 of the paper: under- / over-realization, else mixed."""
    g, p = gp
    d = [pi - gi for gi, pi in zip(g, p)]
    if all(x <= 0 for x in d) and any(x < 0 for x in d):
        return "under"
    if all(x >= 0 for x in d) and any(x > 0 for x in d):
        return "over"
    return "mixed"


def main():
    gt = load_gt()
    covered = {k.split("/")[-1]: bool(v["gt_struct_in_train"])
               for k, v in json.load(io.open(FLAGS, encoding="utf-8"))["flags"].items()}

    out = {"note": "reference = GT_true (official builder); universe per model "
                   "= GT-usable samples with a generation label",
           "models": {}}
    for m in MODELS:
        preds = load_preds(GEN[m], m)
        valid, n_inv = [], 0
        for uid, g in gt.items():
            if uid not in preds:
                continue
            p = preds[uid]
            if p is None:
                n_inv += 1
            else:
                valid.append((g, p))
        n = len(valid)
        hits = sum(1 for g, p in valid if g == p)
        universe = n + n_inv
        comp = [round(sum(1 for g, p in valid if g[i] == p[i]) / n, 4)
                for i in range(3)]
        ctopo = {}
        for lv in ("C0", "C12", "C3p"):
            sub = [(g, p) for g, p in valid if ctopo_layer(g) == lv]
            ctopo[lv] = round(sum(1 for g, p in sub if g == p) / len(sub), 4) \
                if sub else None
        mism = [(g, p) for g, p in valid if g != p]
        dir_counts = {"under": 0, "over": 0, "mixed": 0}
        for g, p in mism:
            dir_counts[direction((g, p))] += 1
        nm = len(mism)
        fig2b = {k: (round(v / nm, 4) if nm else None) for k, v in dir_counts.items()}
        fig2b["n_mismatch"] = nm

        chi = lambda b: b[0] - b[1] + b[2]
        chi_fid = round(sum(1 for g, p in valid if chi(g) == chi(p)) / n, 4)
        chi_inv = sum(1 for g, p in mism if chi(g) == chi(p))
        multi = [(g, p) for g, p in mism
                 if sum(1 for i in range(3) if g[i] != p[i]) >= 2]
        multi_dchi = sum(1 for g, p in multi if chi(g) != chi(p))

        fig3a, fig3b = {}, {}
        covered_pairs = []
        for uid, g in gt.items():
            p = preds.get(uid)
            if p is None:
                continue
            covered_pairs.append((g, p, covered.get(uid)))
        for cov, tag in ((True, "covered"), (False, "novel")):
            sub = [(g, p) for g, p, c in covered_pairs if c is cov]
            k = sum(1 for g, p in sub if g == p)
            fig3a[tag] = {"n": len(sub),
                          "fid": round(k / len(sub), 4) if sub else None}
        for lv in ("C0", "C12", "C3p"):
            row = {}
            for cov, tag in ((True, "covered"), (False, "novel")):
                sub = [(g, p) for g, p, c in covered_pairs
                       if c is cov and ctopo_layer(g) == lv]
                k = sum(1 for g, p in sub if g == p)
                row[tag] = {"n": len(sub),
                            "fid": round(k / len(sub), 4) if sub else None}
            fig3b[lv] = row

        out["models"][m] = {
            "table1": {"n_valid": n, "n_invalid": n_inv,
                       "invalid_rate": round(n_inv / universe, 4),
                       "beta_acc": comp,
                       "sig_fidelity": round(hits / n, 4),
                       # Sig+Inv as reported in the paper: signature fidelity
                       # (4-decimal) scaled by the valid fraction
                       "sig_inv_fidelity": round(round(hits / n, 4) * n / universe, 4)},
            "fig2a_ctopo_fidelity": ctopo,
            "fig2b_direction": fig2b,
            "fig3a_covered_novel": fig3a,
            "fig3b_ctopo_covered_novel": fig3b,
            "table6_chi_fidelity": chi_fid,
            "table7_chi_on_errors": {
                "n_sig_errors": nm,
                "chi_invisible": chi_inv,
                "chi_invisible_pct": round(chi_inv / nm, 4) if nm else None,
                "multi_component": len(multi),
                "multi_component_pct": round(len(multi) / nm, 4) if nm else None,
                "multi_component_dchi_neq0": multi_dchi},
        }
        t1 = out["models"][m]["table1"]
        print(f"{m:<12} valid={n} invalid={n_inv} "
              f"inv_rate={t1['invalid_rate']} sig={t1['sig_fidelity']} "
              f"sig_inv={t1['sig_inv_fidelity']}")

    frozen = json.load(io.open(FROZEN, encoding="utf-8"))
    assert out == frozen, (
        "rq1 aggregate mismatch:\n" + json.dumps(
            {m: {"recomputed": out["models"][m], "frozen": frozen["models"][m]}
             for m in MODELS if out["models"][m] != frozen["models"][m]},
            indent=1))
    print("\nsanity OK: Table 1, Figures 2(a,b), 3(a,b) and appendix Tables 6-7 "
          "exactly match the frozen paper numbers")
    print("frozen ->", FROZEN)


if __name__ == "__main__":
    main()
