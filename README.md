# MGHDA

Reproducible Study 2 and Study 3 experiments for granularity-dependent
hallucination and visual-evidence consistency in medical multimodal models on
chest X-rays.

- Study 2 is the G1/G2 single-claim verification experiment defined in
  [`docs/study2-prompt-construction.md`](docs/study2-prompt-construction.md).
- Study 3 is the independent explicit anatomical-finding multi-select
  experiment defined in
  [`docs/study3-multiselect-construction.md`](docs/study3-multiselect-construction.md).

Study 3 reuses the G1/G2 taxonomy, but it has separate code, configuration,
item IDs, generated data, transfer manifests, inference outputs, scoring, and
figures. Study 2 artifacts are not valid Study 3 inputs.

## Repository layout

```text
configs/                 tracked, non-secret configuration examples
docs/                    experiment specification and schemas
scripts/                 ordered local-preparation and remote-run entrypoints
src/ghm/data/            Chest ImaGenome audit, parsing, and image linking
src/ghm/granularity/     optimized G1/G2 item construction
src/ghm/prompts/         separated model-input/eval-metadata prompt builder
src/ghm/inference/       MedGemma runner
src/ghm/evaluation/      parsing, scoring, validation, and visualization
src/ghm/migration/       preflight and direct-transfer utilities
src/ghm/study3/          isolated Study 3 construction and evaluation
scripts/study3/          isolated local and remote Study 3 entrypoints
tests/                   synthetic tests only
```

## Study 3 local data preparation

Study 3 reads the already normalized Chest ImaGenome tables and only keeps
MIMIC-CXR-JPG images that already exist under `files/`. It never downloads
images. From PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$env:MGHDA_DATA_ROOT = 'C:\Users\24540\data'
$env:MGHDA_PYTHON = 'C:\path\to\python.exe' # omit when .venv exists
.\scripts\study3\00_audit_candidates.ps1
.\scripts\study3\01_build_items.ps1
.\scripts\study3\02_build_prompts.ps1
.\scripts\study3\03_prepare_transfer_manifest.ps1
```

Equivalent Linux/Git Bash commands are:

```bash
cp configs/study3_medgemma.env.example configs/study3_medgemma.env
bash scripts/study3/00_audit_candidates.sh
bash scripts/study3/01_build_items.sh
bash scripts/study3/02_build_prompts.sh
bash scripts/study3/03_prepare_transfer_manifest.sh
```

The reference target is 1,800 prompts each for G1 and G2: 1,000 natural-option
prompts plus 800 controlled K=2/3/4/5 prompts. Candidate shortages are reported
and are never filled with unmentioned findings.

Study 3 private artifacts are isolated under `interim/study3`,
`processed/study3`, and `outputs/study3`. The transfer list is:

```text
outputs/study3/transfer/study3_files_from.txt
```

Upload from the existing data root without duplicating the image tree:

```bash
rsync -av --partial \
  --files-from=outputs/study3/transfer/study3_files_from.txt \
  /path/to/data/ user@host:/path/to/data/
```

The code checkout on the host must match the Git commit recorded in
`study3_transfer_summary.json`.

## Study 3 remote inference

On the GPU host, configure `configs/study3_medgemma.env`, then run:

```bash
bash scripts/study3/04_verify_and_preflight.sh
bash scripts/study3/run_smoke.sh
bash scripts/study3/run_full.sh
```

`run_full.sh` performs resumable MedGemma inference, multi-select scoring,
structural validation, aggregate reporting, and figures. Study 3 defaults to
`RUN_NAME=medgemma_study3_multiselect_v1`; its scripts never source the Study 2
environment file.

## Study 2 current Windows data machine

The data root is the existing `C:\Users\24540\data` directory. It directly
contains `interim`, `processed`, `files`, `outputs`, and `raw`; scripts must not
append another `data` component and do not require an `MGHDA-private` directory.

From PowerShell, after creating a Python 3.10/3.11 environment with the project
dependencies, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$env:MGHDA_PYTHON = "C:\path\to\python.exe" # omit when .venv exists
.\scripts\00_audit_schema.ps1
.\scripts\01_build_items.ps1
.\scripts\02_build_prompts.ps1
.\scripts\03_prepare_transfer_manifest.ps1
```

These commands write only inside the existing data root. The final command does
not copy or move images; it writes `outputs\transfer\study2_files_from.txt`, a
SHA256 manifest, and an aggregate summary. All paths in generated prompt JSONL
are portable and relative to the data root, beginning with `files/`.

## Study 2 trusted data machine

On Linux or Git Bash, copy the environment example and use the equivalent shell
entrypoints:

```bash
bash scripts/00_audit_schema.sh
bash scripts/01_build_items.sh
bash scripts/02_build_prompts.sh
bash scripts/prepare_inference_bundle.sh
```

The linked Study 2 set defaults to 960 model inputs: 480 each for G1 and G2.
Sampling keeps positive/negative claim pairs and balances affirmed, negated, and
not-enough-evidence pairs, using only images already present under `files/`.
If candidate construction already completed but linking was interrupted, resume
without rebuilding candidates:

```bash
bash scripts/01b_link_and_sample_items.sh
```

Upload directly from the existing data root with the generated file list. No
second private-data directory is needed:

```bash
rsync -av --partial --files-from=outputs/transfer/study2_files_from.txt \
  /path/to/data/ user@new-host:/path/to/data/
```

Images, prompts, eval metadata, checksums, and patient-linked outputs remain
restricted and must never be committed to GitHub.

## Study 2 remote GPU machine

Point `MGHDA_DATA_ROOT` at the transferred data root, configure the local
MedGemma 4B IT path, then run:

```bash
bash scripts/verify_inference_bundle.sh
bash scripts/preflight_medgemma.sh
bash scripts/dry_run_medgemma_inputs.sh
bash scripts/run_medgemma_smoke.sh
bash scripts/run_medgemma_full.sh
```

The smoke and full wrappers each run inference, scoring, and structural
validation with the same exported run name; the full wrapper also creates the
aggregate figures. Standalone score, validation, and visualization scripts
remain available for re-analysis.

The runner checkpoints each item and supports resume. Full row-level outputs
remain private; only aggregate summaries without identifiers or paths are safe
to share.

## Reference environment

- Python 3.10 or 3.11
- PyTorch 2.5.1 with CUDA 12.1 (install separately for the platform)
- Transformers 4.50.3
- Accelerate 1.13.0
- Pillow, PyArrow, Matplotlib

The preflight command checks the actual platform and never downloads a model.
