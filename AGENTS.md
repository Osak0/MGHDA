# AGENTS.md

## Project Overview

This project studies granularity-dependent hallucination in medical multimodal
large language models using chest X-ray datasets.

## Current Taxonomy

- `G1_finding_existence`: image-level finding existence.
- `G2_anatomical_localization`: anatomy-specific finding existence or
  anatomical localization.
- `G3a_contrastive_finding_anatomy_binding`: contrastive finding-anatomy
  binding.
- `G3b_modifier_characterization`: modifier characterization with weak/silver
  evidence.
- `G3c_temporal_or_comparison_relation`: reserved temporal or comparison
  relation analysis.
- `G4_structured_synthesis_proxy`: structured synthesis proxy.

## Hallucination Labels

- `H1_evidence_contradicted`: evidence-contradicted hallucination.
- `H2_evidence_unsupported`: evidence-unsupported hallucination.

## Research Rules

- Do not redefine the taxonomy unless explicitly instructed.
- Do not convert missing labels into `No`.
- Treat `Unknown` and `Conflict` as separate states.
- Do not treat Chest ImaGenome anatomical bboxes as lesion segmentation.
- Do not use `bbox_name=False` evidence for G2 anatomical localization.
- Do not implement model training yet.
- Do not download models yet.
- Do not process real restricted data until dataset access and audit are ready.
- Do not commit restricted medical images, reports, or patient-linked outputs.

## Engineering Rules

- Prefer small, testable modules.
- Keep generated samples traceable with `evidence_sources` and
  `source_assertions`.
- All scripts should support config files once implementation begins.
- Use JSONL for model inputs and outputs.
- Use parquet for large processed tables.
- Add tests for parsing and scoring logic when implementation begins.
- Follow `docs/data-schema.md` and `docs/granularity-mapping.md` as source of
  truth.

## Restricted Data Policy

This project may use restricted PhysioNet datasets such as MIMIC-CXR, MIMIC-CXR-JPG, Chest ImaGenome, and RadGraph.

Do not read, open, summarize, print, copy, upload, transform, or inspect any real restricted data files.

Restricted data includes:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/pilot/`
- real medical images
- real radiology reports
- real scene graph JSON files
- real RadGraph annotations
- any patient-linked outputs

Only use:

- synthetic fixtures under `tests/fixtures/`
- toy examples under `data/fixtures/`
- schema documents under `docs/`
- sanitized aggregate audit outputs that contain counts only

When writing code, use paths from config files or environment variables. Do not hard-code real dataset paths.

When debugging, do not print patient_id, study_id, image_id, report text, raw phrases, or image contents. Print only:

- counts
- column names
- schema keys
- aggregate distributions
- first-level file names if sanitized

If real data processing is needed, write scripts only. The user will run them locally and share only sanitized logs.
