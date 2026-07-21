# Study 2/3 v2 Clean Deployment

## Local preparation

Run only on the trusted data machine:

```bash
bash scripts/02_build_prompts.sh
bash scripts/study3/00_audit_candidates.sh
bash scripts/study3/01_build_items.sh
bash scripts/study3/02_build_prompts.sh
pytest -q
```

Commit the code. A private package is intentionally tied to that clean commit:

```bash
bash scripts/package_release.sh
```

The packager rejects missing images, unsafe/absolute paths, answer leakage,
duplicate IDs, mismatched input/metadata IDs, incorrect Study 2 ablation
pairing, non-960 Study 2 full inputs, non-120 ablation inputs, non-3,600 Study
3 inputs, non-40 smoke inputs, and incomplete Study 3 four-question sets.

## Upload

Create `incoming/<commit>/` in the persistent remote workspace and upload:

```text
MGHDA-study2v1v2-study3v2-<commit>.tar.gz
MGHDA-study2v1v2-study3v2-<commit>.tar.gz.sha256
MGHDA-private-study2v1v2-study3v2-<commit>.tar.gz
MGHDA-private-study2v1v2-study3v2-<commit>.tar.gz.sha256
```

Do not delete the old remote directories until all four files are present.

## Clean replacement

From the old checkout:

```bash
bash scripts/deploy/replace_remote.sh /persistent/workspace <commit>
```

The script:

1. verifies both outer SHA256 sidecars;
2. rejects traversal paths, links, devices, and duplicate tar members;
3. resolves deletion targets and accepts only:
   `code/MGHDA`, `files`, `processed`, `outputs`, `interim`, and `raw`;
4. preserves `models`, `envs`, and `incoming`;
5. extracts into clean directories;
6. verifies the private internal SHA256 manifest.

Keep `incoming/<commit>/` until Study 2 and Study 3 smoke runs succeed.

## Existing environment and models

Do not recreate the environment. Activate it and minimally upgrade only if
preflight reports an old dependency:

```bash
python -m pip install --upgrade 'transformers>=4.57.0' 'accelerate>=1.13.0' modelscope
```

Download a missing approved model explicitly:

```bash
bash scripts/models/download_modelscope.sh \
  google/medgemma-1.5-4b-it \
  /persistent/workspace/models/google/medgemma-1.5-4b-it
```

Copy and edit model paths, then validate all three:

```bash
cp configs/models.env.example configs/models.env
bash scripts/preflight_all_models.sh
pytest -q
bash scripts/dry_run_all_inputs.sh
```

Preflight checks CUDA, Python, packages, disk root, and `config.json` in all
three local model directories. It never downloads a model.

## Run matrix

For each model, export canonical `MODEL_NAME` and local `MODEL_PATH`, then use
unique run names:

```bash
STUDY2_SET=full_v1 RUN_PHASE=ablation RUN_NAME=<model>_study2_v1_full bash scripts/study2/run.sh
STUDY2_SET=full_v2 RUN_PHASE=ablation RUN_NAME=<model>_study2_v2_full bash scripts/study2/run.sh
V1_RUN_NAME=<model>_study2_v1_full \
V2_RUN_NAME=<model>_study2_v2_full \
  bash scripts/study2/compare_ablation.sh
STUDY2_SET=full_v1 RUN_PHASE=full RUN_NAME=<model>_study2_v1_full bash scripts/study2/run.sh
STUDY2_SET=full_v2 RUN_PHASE=full RUN_NAME=<model>_study2_v2_full bash scripts/study2/run.sh

RUN_NAME=<model>_study3_v2_smoke bash scripts/study3/run_smoke.sh
RUN_NAME=<model>_study3_v2_full bash scripts/study3/run_full.sh
```

Repeat for:

```text
google/medgemma-4b-it
google/medgemma-1.5-4b-it
Qwen/Qwen3-VL-8B-Instruct
```

After `configs/models.env` is configured, the phase driver runs exactly this
matrix without crossing the two review gates:

```bash
bash scripts/run_three_models_phase.sh study2-ablation
bash scripts/run_three_models_phase.sh study2-full
bash scripts/run_three_models_phase.sh study3-smoke
bash scripts/run_three_models_phase.sh study3-full
```

Review the ablation summaries before `study2-full`, and require all three smoke
validations to pass before `study3-full`.

Do not reuse a run name across a model, Study, template, or smoke/full setting.
