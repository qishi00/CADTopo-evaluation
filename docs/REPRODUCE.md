# Reproduction Guide

This guide reproduces the paper's headline results from the shipped data.
Commands are run from the repository root.

## 0. Setup

```bash
pip install -r requirements.txt
# Optional, only needed to rebuild STEP files from DeepCAD sequences:
# place the DeepCAD repo at external/DeepCAD (provides cadlib + create_CAD)
```

No external data is required for the steps below; everything runs on the labels and frozen artifacts in `data/`.

## 1. Topology labeling (used for any new generations)

```bash
python evaluation/label_steps.py /path/to/generated/steps --out my_labels.jsonl
```

Each output row carries the watertight Betti signature `betti = (β0, β1, β2)` plus validity flags; `ok=false` marks invalid solids (the F0 class in the paper).

To verify that the labeling pipeline reproduces the paper's per-sample labels from raw STEP files:

```bash
python check_labeling.py
```

This re-labels the 46 stratified sample STEP files in `examples/steps_skexgen_sample/` (covering every Betti-signature bucket: no-hole, through-holes, cavities, multi-component solids, and invalid cases) and asserts exact agreement with `data/labels/skexgen_labels_v1.jsonl`. `run_all_checks.py` includes this as its first check.

## 2. RQ2 — complexity-standardized covered–novel gap (Table 7)

```bash
python rq2/standardize_gap.py
```

The script recomputes, per model, the raw and standardized covered–novel fidelity gap over a 3×3 stratification (C_topo layers × command-length
terciles, L ≤ 6 / 7–14 / ≥ 15), reweighted to the novel group's cell distribution, with 95% sample-level bootstrap CIs (1000 resamples). Built-in asserts verify equality with the frozen numbers behind Table 7.

## 3. RQ3-N — topology-directed intervention (Sec. 4.3)

```bash
python rq3/aggregate_rq3n.py
```

Recomputes the five-class intervention outcome (hit / partial / ignore / wrong_dir / other / invalid) for the 600 exact-baseline bases (300 add-hole, 300 remove-hole) of the primary protocol, and asserts equality with the frozen result. The per-sample labels are in `data/labels/rq3n/`; the task pack (prompts, target signatures, selection seed) is `data/prompts/rq3n_tasks_server.jsonl`.

## 4. Distractor control (Sec. 4.3)

```bash
python rq3/rq3_conditional_on_A_v1.py
```

Reproduces the topology-neutral distractor stability numbers (99.5% / 99.7% / 98.1%) under the 931-base protocol, plus the v2 add/remove breakdowns.

## 5. Counterfactual prompts (Appendix C)

`rq3/t4c_counterfactual_prompts_v2.py` regenerates the expert-level prompt pack from the public Text2CAD CSV and the shipped GT annotation file; the generated pack is already included as `data/prompts/t4c_prompts_v2_expert.jsonl`.

## 6. Appendix F — vectorization-induced signature changes

`data/results/appF_762_betti_pairs_v1.json` lists the 762 test references whose Betti signature differs between the unquantized CAD JSON and the official 8-bit quantized vector, with both signatures and the mismatch direction (over / under / mixed).
