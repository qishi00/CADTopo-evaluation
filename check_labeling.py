# -*- coding: utf-8 -*-
"""check_labeling.py — verify that the shipped labeling pipeline reproduces
the paper's per-sample Betti labels from raw STEP files.

Re-runs evaluation/label_steps.py on the 46 sample STEP files in
examples/steps_skexgen_sample/ (a stratified subset of the SkexGen test-set
reconstructions, chosen to cover every Betti-signature bucket: no-hole,
through-holes, cavities, multi-component solids, and invalid/non-watertight
cases) and asserts that ok / betti / error match the shipped labels in
data/labels/skexgen_labels_v1.jsonl exactly.

Exits 0 on full match, 1 otherwise.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "examples" / "steps_skexgen_sample"
SHIPPED = ROOT / "data" / "labels" / "skexgen_labels_v1.jsonl"


def main():
    shipped = {}
    for line in SHIPPED.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("cond") is None and r.get("k") is None:
            shipped[Path(r["file"]).name] = r

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "relabel.jsonl"
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(ROOT / "evaluation" / "label_steps.py"),
             str(SAMPLE_DIR), "--out", str(out)],
            cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print("FAIL: label_steps.py exited with code", proc.returncode)
            sys.exit(1)

        n_match = 0
        mismatches = []
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            name = Path(r["file"]).name
            s = shipped.get(name)
            if s is None:
                mismatches.append((name, "not found in shipped labels"))
                continue
            if (r["ok"], r["betti"], r["error"]) == (s["ok"], s["betti"], s["error"]):
                n_match += 1
            else:
                mismatches.append((name, f"shipped betti={s['betti']} ok={s['ok']} "
                                         f"error={s['error']} | recomputed betti={r['betti']} "
                                         f"ok={r['ok']} error={r['error']}"))

    n_total = n_match + len(mismatches)
    print(f"relabeled {n_match}/{n_total} sample STEP files "
          f"(stratified across all Betti buckets)")
    if mismatches:
        for name, msg in mismatches:
            print(f"  MISMATCH {name}: {msg}")
        print(f"FAIL: {len(mismatches)} mismatch(es)")
        sys.exit(1)
    print("PASS: ok / betti / error reproduce the shipped labels exactly")


if __name__ == "__main__":
    main()
