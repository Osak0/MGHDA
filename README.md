# MGHDA

Research on granularity-dependent hallucination in medical multimodal large
language models, with a first-stage focus on chest X-ray data.

## Current Phase

This repository is currently in the pre-data engineering setup phase. The
project has not yet started real dataset parsing, model inference, model
download, or training.

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
  -> model-ready items
  -> prompts
  -> model outputs
  -> scored outputs
  -> analysis
```

## Data Safety

Do not commit raw medical images, radiology reports, restricted-access dataset
files, or patient-linked outputs to Git. Public fixtures should use toy examples
or fully de-identified synthetic records only.

## Training Status

Training, LoRA, SFT, and DPO are reserved for later stages. They should only be
considered after data parsing, schema audit, and pilot inference evaluation are
stable.

