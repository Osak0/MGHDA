# MGHDA

Reproducible Study 2/3 experiments for granularity-dependent hallucination and
visual-evidence consistency in chest X-ray multimodal models.

All real images, prompts, evaluation metadata, model weights, checkpoints, and
row-level outputs are restricted. They must remain outside Git.

## Current experiments

- Study 2 keeps `claim_verification_abc_v1` and adds the definition-only
  `claim_verification_abc_definitions_v2`.
- Study 2 uses one fixed 120-item paired v1/v2 ablation
  (`G1/G2 × evidence state × claim polarity`, 10 per cell), a 960-item full v1
  run, and a 960-item full v2 run.
- Study 3 is `study3_multiselect_v2`. Every option set has four questions:
  `state/evidence × present/absent`.
- Study 3 uses 250 natural and 50 controlled anchors per granularity, producing
  1,800 records for G1 and 1,800 for G2.
- The same inputs are evaluated with:
  `google/medgemma-4b-it`, `google/medgemma-1.5-4b-it`, and
  `Qwen/Qwen3-VL-8B-Instruct`.

The authoritative protocols are
[`docs/study2-prompt-construction.md`](docs/study2-prompt-construction.md) and
[`docs/study3-multiselect-construction.md`](docs/study3-multiselect-construction.md).
The complete deployment checklist is
[`docs/study2-study3-v2-deployment.md`](docs/study2-study3-v2-deployment.md).

## Trusted data-machine preparation

`MGHDA_DATA_ROOT` is the existing data root that directly contains `files/`,
`interim/`, `processed/`, `outputs/`, and `raw/`. Do not create an additional
`MGHDA-private` directory.

```bash
cp configs/study2_medgemma.env.example configs/study2_medgemma.env
# Edit only MGHDA_DATA_ROOT for local preparation.

bash scripts/02_build_prompts.sh
bash scripts/study3/00_audit_candidates.sh
bash scripts/study3/01_build_items.sh
bash scripts/study3/02_build_prompts.sh
pytest -q
```

The generated private layout is:

```text
files/
processed/study2/v1/
processed/study2/v2/
processed/study2/ablation/v1/
processed/study2/ablation/v2/
processed/study3/v2/
outputs/transfer/
```

After committing a clean code revision, create both upload archives:

```bash
bash scripts/package_release.sh
```

This creates a code archive and a private archive under
`artifacts/release-<commit>/`, each with an outer SHA256 sidecar. The private
archive contains only required versioned inputs, Study 3 items, deduplicated
images, an aggregate manifest, and an internal checksum.

## Remote replacement and inference

Upload the four archive/sidecar files to `incoming/<commit>/`. Then run the
replacement script from the old checkout:

```bash
bash scripts/deploy/replace_remote.sh /persistent/workspace <commit>
```

It verifies both uploads before deleting the exact allowlisted old code/data
directories. It preserves `models/`, `envs/`, and `incoming/`.

Activate the existing environment, copy `configs/models.env.example` to the
ignored `configs/models.env`, and run:

```bash
bash scripts/preflight_all_models.sh
pytest -q
bash scripts/dry_run_all_inputs.sh
```

For each model, export its canonical ID and local directory. Example:

```bash
export MODEL_NAME='Qwen/Qwen3-VL-8B-Instruct'
export MODEL_PATH='/persistent/workspace/models/Qwen/Qwen3-VL-8B-Instruct'
```

Study 2:

```bash
STUDY2_SET=full_v1 RUN_PHASE=ablation RUN_NAME=qwen3vl8b_study2_v1_full \
  bash scripts/study2/run.sh
STUDY2_SET=full_v2 RUN_PHASE=ablation RUN_NAME=qwen3vl8b_study2_v2_full \
  bash scripts/study2/run.sh
V1_RUN_NAME=qwen3vl8b_study2_v1_full \
V2_RUN_NAME=qwen3vl8b_study2_v2_full \
  bash scripts/study2/compare_ablation.sh
STUDY2_SET=full_v1 RUN_PHASE=full RUN_NAME=qwen3vl8b_study2_v1_full \
  bash scripts/study2/run.sh
STUDY2_SET=full_v2 RUN_PHASE=full RUN_NAME=qwen3vl8b_study2_v2_full \
  bash scripts/study2/run.sh
```

The same staged matrix for all three configured models is:

```bash
bash scripts/run_three_models_phase.sh study2-ablation
# Inspect the three paired comparison summaries before continuing.
bash scripts/run_three_models_phase.sh study2-full
bash scripts/run_three_models_phase.sh study3-smoke
# Inspect all smoke validations before continuing.
bash scripts/run_three_models_phase.sh study3-full
```

Study 3:

```bash
RUN_NAME=qwen3vl8b_study3_v2_smoke bash scripts/study3/run_smoke.sh
RUN_NAME=qwen3vl8b_study3_v2_full bash scripts/study3/run_full.sh
```

Use a different `RUN_NAME` for every model and Study 2 template. The ablation
and full phases deliberately share that run's checkpoint, so the first 120
successful items are not inferred twice. Study 3 smoke/full use different run
names. Resume skips successful items and retries failed items.

## Environment

- Python 3.10 or 3.11
- CUDA-enabled PyTorch appropriate for the remote driver
- Transformers 4.57.0 or newer
- Accelerate 1.13.0 or newer
- Pillow, PyArrow, Matplotlib, ModelScope, Pytest

The preflight never downloads models. Model downloads are explicit:

```bash
bash scripts/models/download_modelscope.sh \
  google/medgemma-1.5-4b-it \
  /persistent/workspace/models/google/medgemma-1.5-4b-it
```
