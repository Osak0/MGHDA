# MGHDA

Reproducible Study 2 experiments for granularity-dependent hallucination in
medical multimodal models on chest X-rays.

The current runnable scope is the optimized G1/G2 claim-verification experiment
defined in [`docs/study2-prompt-construction.md`](docs/study2-prompt-construction.md).
Every prompt uses the same answer space: Supported, Contradicted, or Not enough
evidence. G2 missing-evidence items use same-bbox vocabulary and bbox-quality
gating.

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
tests/                   synthetic tests only
```

## Current Windows data machine

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

## Trusted data machine

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

## Remote GPU machine

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
