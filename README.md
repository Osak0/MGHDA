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
src/ghm/migration/       preflight and private-bundle utilities
tests/                   synthetic tests only
```

## Configure

```bash
cp configs/study2_medgemma.env.example configs/study2_medgemma.env
# Edit the private paths in configs/study2_medgemma.env.
```

The private config, `data/`, `outputs/`, model weights, images, prompt JSONL,
eval metadata, and patient-linked outputs must never be committed.

## Trusted data machine

Run the aggregate bbox audit, rebuild optimized items, build prompts, and create
the minimum private inference bundle:

```bash
bash scripts/00_audit_schema.sh
bash scripts/01_build_items.sh
bash scripts/02_build_prompts.sh
bash scripts/prepare_inference_bundle.sh /path/to/private/study2-bundle
```

The bundle contains only the required images, prompt JSONL, eval metadata, an
aggregate manifest, and a private SHA256 file. Upload it with a resumable private
transport such as `rsync`; never upload it to GitHub.

## Remote GPU machine

Point `MGHDA_DATA_ROOT` at the verified bundle root, configure the local
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
