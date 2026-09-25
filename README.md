# CAD-TopoBench: Benchmark Code and Data

Code and experimental data for **CAD-TopoBench**, a topology-oriented benchmark for CAD generative models (ICLR 2027 submission, double-blind).

CAD-TopoBench evaluates generative CAD models along four dimensions that go beyond geometry: **topology fidelity** (exact Betti-signature match against the reference), **geometry–topology consistency**, **topology diversity**, and **controllability under topology-directed interventions**. This repository contains the reference topology extractor, the structural-fingerprint coverage pipeline, the counterfactual intervention prompts and aggregation code, and all precomputed per-sample labels required to reproduce the paper's tables.

## Repository layout

```
cad_topobench/        Core package
  reference.py          solid_betti(): STEP -> watertight Betti signature (reference v1.0)
  topology.py, geometry.py, metrics.py, kendall.py, pareto.py, entropy.py, bootstrap.py
  paths.py              all locations resolve via CAD_TOPOBENCH_ROOT (default: repo root)
evaluation/
  label_steps.py        CLI: label every .step in a directory with its Betti signature
rq2/
  fingerprint.py                 structural fingerprint over the DeepCAD command vocabulary
  build_model_specific_coverage.py   model-specific covered/novel split (union of training branches)
  standardize_gap.py             complexity-standardized covered-novel gap (Table 7), with sanity asserts
rq3/
  t4c_counterfactual_prompts_v2.py   counterfactual prompt construction (add-hole / remove-hole / distractor)
  rq3_conditional_on_A_v1.py         intervention aggregation conditioned on exact baseline fidelity
data/
  labels/               per-sample Betti labels for all five models + GT references (jsonl)
  coverage/             model-specific coverage splits (covered/novel uid lists)
  prompts/              counterfactual prompt sets
  audit/                frozen intermediate artifacts (fingerprints, per-model RQ2b JSONs)
  results/              frozen headline results (Table 7 gap, RQ3 table, Appendix F pairs)
external/             public datasets, NOT shipped (see "External data" below)
docs/REPRODUCE.md     step-by-step reproduction guide
```

## Quickstart

```bash
pip install -r requirements.txt

# Label your own generated STEP files:
python evaluation/label_steps.py /path/to/steps

# Reproduce Table 7 (complexity-standardized covered-novel gap):
python rq2/standardize_gap.py

# Reproduce the RQ3-N intervention table (Sec. 4.3 strict-hit rates):
python rq3/aggregate_rq3n.py

# Distractor control (topology-neutral stability, Sec. 4.3):
python rq3/rq3_conditional_on_A_v1.py
```

Both aggregation scripts carry built-in sanity assertions against the frozen results in `data/`; a successful run means every number matches the paper.

## External data (not shipped)

The benchmark builds on public datasets and model checkpoints; download them into `external/`:

| Path | Source |
|---|---|
| `external/DeepCAD` | DeepCAD official repo (Wu et al., 2021) — provides `cadlib` and the `create_CAD` reconstruction chain used by the reference extractor |
| `external/cad_json` | DeepCAD JSON sequences (test split) |
| `external/cad_vec` | DeepCAD 8-bit quantized vectors (test split) |
| `external/train_val_test_split.json` | DeepCAD official split |
| `external/text2cad_v1.0.csv` | Text2CAD v1.0 prompt table (expert level used for counterfactual prompts) |
| model checkpoints | DeepCAD, SkexGen, Text2CAD, CAD-Coder, Cadrille-SFT official releases |

## Environment

- Python 3.10
- cadquery (OCp/OCC bindings), numpy, scipy, trimesh, h5py
- Importing `cad_topobench.geometry`/`reference` requires the DeepCAD repo on
  `sys.path`; `cad_topobench.paths` expects it at `external/DeepCAD`.

## Notes

- All bootstrap confidence intervals are 95%, sample-level, 1000 resamples.
- `data/results/appF_762_betti_pairs_v1.json` lists the 762 test references whose
  Betti signature changes under the official 8-bit vectorization (Appendix F).
- Random seeds are fixed in each script; aggregation results are deterministic.
