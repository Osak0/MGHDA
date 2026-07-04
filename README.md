# MGHDA

Research on granularity-dependent hallucination in medical multimodal large
language models, with a first-stage focus on chest X-ray data.

## Current Phase

This repository is currently focused on the Study 2 G1/G2 claim-verification
pilot. Training, model downloads, and unrestricted real-data inspection remain
out of scope for this repo workflow.

## Project Goal

The project studies whether medical multimodal large language models hallucinate
differently across clinical-semantic granularity levels. The initial benchmark
will derive multi-granularity experimental items from MIMIC-CXR /
MIMIC-CXR-JPG, Chest ImaGenome, and RadGraph after the datasets are available
and audited.

## Planned Pipeline

```text
raw data
  -> intermediate tables
  -> Study 2 model-ready claim-verification items
  -> prompts
  -> model outputs
  -> scored outputs
  -> analysis
```

Study 2 uses one ABC answer space:

```text
A. Supported
B. Contradicted
C. Not enough evidence
```

G1 and G2 items are regenerated from intermediate Chest ImaGenome tables, then
linked to already-downloaded MIMIC-CXR-JPG files with `--link-only-existing`.

## Data Safety

Do not commit raw medical images, radiology reports, restricted-access dataset
files, or patient-linked outputs to Git. Public fixtures should use toy examples
or fully de-identified synthetic records only.

## Training Status

Training, LoRA, SFT, and DPO are reserved for later stages. They should only be
considered after data parsing, schema audit, and pilot inference evaluation are
stable.

## pilot experiment
torch: 2.5.1+cu121
cuda: True
torch cuda: 12.1
transformers: 4.50.3
accelerate: 1.13.0
pillow ok
