# -*- coding: utf-8 -*-
"""run_all_checks.py — one-command reproducibility check for CAD-TopoBench.

Runs every headline-number pipeline against the shipped data and reports
PASS / FAIL per check. Exits 0 only if all checks pass.

Usage (from the repository root, any Python >= 3.10 with the packages in
requirements.txt installed):

    python run_all_checks.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (name, script, what a PASS proves)
CHECKS = [
    ("Labeling  STEP -> Betti signature (46 sample files)",
     "check_labeling.py",
     "the topology labeling pipeline reproduces the shipped per-sample "
     "ok / betti / error labels from raw STEP files"),
    ("RQ2  Fig 3(c)  standardized covered-novel gap",
     "rq2/standardize_gap.py",
     "the five per-model standardized gaps + 95% CIs match the frozen numbers"),
    ("RQ3-N  intervention five-class counts (Sec. 4.3)",
     "rq3/aggregate_rq3n.py",
     "the hit/ignore/... counts over the 600 exact-baseline bases match"),
    ("RQ3    distractor stability + add/remove breakdown",
     "rq3/rq3_conditional_on_A_v1.py",
     "the 99.5% / 99.7% / 98.1% distractor numbers and v2 breakdowns reproduce"),
]


def run_check(name, script):
    print(f"\n=== CHECK: {name}")
    print(f"    $ python {script}")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / script)],
        cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode == 0:
        print(f"    PASS   ({elapsed:.1f}s)")
        return True
    print(f"    FAIL   ({elapsed:.1f}s, exit code {proc.returncode})")
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
    for line in tail:
        print("    | " + line)
    return False


def main():
    print("CAD-TopoBench reproducibility check")
    print(f"repo: {ROOT}")
    print(f"python: {sys.executable}")
    results = [(name, run_check(name, script)) for name, script, _ in CHECKS]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, _, proves in CHECKS:
        ok = dict((n, r) for n, r in results)[name]
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         proves: {proves}")
    print("=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — the shipped data reproduces the paper's "
              "headline numbers exactly.")
    else:
        print("SOME CHECKS FAILED — see output above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
