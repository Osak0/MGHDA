# Dataset schema

## Introduction

This document defines the unified JSONL schema used after converting raw annotations from MIMIC-CXR, Chest ImaGenome, RadGraph into model-ready and evaluation samples.

---

## Dataset roles

| Dataset | Role in this project | Main evidence type |
| --- | --- | --- |
| MIMIC-CXR | Source dataset containing chest radiographs and radiology reports | image-report backbone |
| MIMIC-CXR-JPG | JPG image version of MIMIC-CXR, easier for VLM inference | model image input |
| Chest ImaGenome | Anatomy-centered scene graph with objects, attributes, bbox, phrases, and optional comparison relations | E1/E2/E3 structured evidence |
| RadGraph | Report-side entity-relation graph extracted from radiology reports | E3 report-structured evidence |

## Identifier schema

### Identifier hierarchy

```text
patient_id / subject_id
    -> study_id
        -> image_id / dicom_id
            -> image_path
```

### Identification fields

| Field | Type | Description |
| --- | --- | --- |
| patient_id | string | Single patient identifier. |
| study_id | string | Single study/report identifier. |
| image_id | string/null | Single image identifier. |
| image_path | image_path | Local path to the MIMIC-CXR-JPG image used as model input. |
| source_dataset | string | The source dataset of the data. |

## Evidence Source Schema

Evidence sources are defined according to the question type. Each question type requires different dataset fields as evidence.

### G1 finding existence questions

Question form:
`Is there evidence of {finding} in this chest X-ray?`

Required evidence:

- finding-level positive / negative assertion.

Dataset fields:

- Chest ImaGenome `attributes`
- Chest ImaGenome `phrases`
- RadGraph OBS-DP / OBS-DA / OBS-U

Evidence sources:

- `E2_structured_label`
- `E3_report_phrase`
- `E3_radgraph`

Answer construction:

- `yes|finding` or OBS-DP -> `Yes`
- `no|finding` or OBS-DA -> `No`
- OBS-U -> `Uncertain`
- missing finding -> `Unknown`, not `No`

### G2 anatomical localization questions

Question form:
`Is there evidence of {finding} in the {anatomy}?`

Required evidence:

- anatomy-finding binding.

Dataset fields:

- Chest ImaGenome valid `bbox_name`
- Chest ImaGenome `attributes`
- Chest ImaGenome `objects` bbox
- RadGraph `located_at(Observation, Anatomy)`

Evidence sources:

- `E1_bbox`
- `E2_anatomy_finding_pair`
- `E3_report_phrase`
- `E3_radgraph`

Answer construction:

- target anatomy has `yes|finding` -> `Yes`
- target anatomy has explicit `no|finding` -> `No`
- `bbox_name=False` -> exclude from G2
- missing anatomy-finding binding -> `Unknown`, not `No`

### G3a contrastive binding / false-premise questions

Question form:
`The {finding} is located in {wrong_anatomy}. Is this statement supported?`

Required evidence:

- true anatomy-finding binding;
- constructed or explicit distractor anatomy.

Dataset fields:

- Chest ImaGenome `bbox_name + attributes`
- RadGraph `located_at`
- constructed distractor anatomy

Evidence sources:

- `E2_anatomy_finding_pair`
- `E3_radgraph`
- `E5_task_context`

Answer construction:

- if statement matches true binding -> `Yes`
- if statement contradicts true binding -> `No`
- if true binding is ambiguous -> exclude or manual review

### G3b modifier questions

Question form:
`Is the {finding} described as {modifier}?`

Required evidence:
    - positive finding assertion;
    - modifier cue or modify relation;
    - phrase-level validation.

Dataset fields:
    - Chest ImaGenome `severity_cues`
    - Chest ImaGenome `temporal_cues`
    - Chest ImaGenome `texture_cues`
    - Chest ImaGenome `comparison_cues`
    - Chest ImaGenome `phrases`
    - RadGraph `modify`

Evidence sources:
    - `E2_modifier_cue`
    - `E3_report_phrase`
    - `E3_radgraph`

Answer construction:
    - positive finding + modifier cue + phrase support -> `Yes`
    - explicit different/conflicting modifier -> `No` or manual review
    - missing modifier -> `Unknown`, not `No`

## Intermediate tables

### 5.1 image_index

Purpose: link MIMIC-CXR-JPG images with MIMIC study IDs, Chest ImaGenome scene graphs, and RadGraph report annotations.

| Field | Type | Description |
| --- | --- | --- |
| `patient_id` | string | Patient identifier, usually MIMIC `subject_id`. |
| `study_id` | string | Study/report identifier. |
| `image_id` | string | Single image identifier, usually equivalent to `dicom_id`. |
| `dicom_id` | string | MIMIC DICOM identifier. |
| `image_path` | string | Local path to the corresponding MIMIC-CXR-JPG image. |
| `view_position` | string/null | AP, PA, LATERAL, etc., if available. |
| `has_chest_imagenome` | bool | Whether this image has Chest ImaGenome scene graph. |
| `has_radgraph` | bool | Whether this study has RadGraph annotation. |

### 5.2 ci_objects

Purpose: store Chest ImaGenome anatomical objects and bounding boxes.

| Field | Type | Description |
| --- | --- | --- |
| `image_id` | string | Chest ImaGenome image ID / MIMIC dicom ID. |
| `study_id` | string | MIMIC study ID. |
| `patient_id` | string | MIMIC subject ID. |
| `bbox_name` | string | Anatomical region name, e.g. `right lung`. |
| `name` | string/null | Human-readable anatomy name. |
| `synsets` | list[string] | UMLS / concept IDs if available. |
| `x1` | float/null | Bbox x1 in resized 224x224 coordinate space. |
| `y1` | float/null | Bbox y1 in resized 224x224 coordinate space. |
| `x2` | float/null | Bbox x2 in resized 224x224 coordinate space. |
| `y2` | float/null | Bbox y2 in resized 224x224 coordinate space. |
| `original_x1` | float/null | Bbox x1 in original image coordinate space. |
| `original_y1` | float/null | Bbox y1 in original image coordinate space. |
| `original_x2` | float/null | Bbox x2 in original image coordinate space. |
| `original_y2` | float/null | Bbox y2 in original image coordinate space. |
| `source_quality` | string | `gold`, `silver`, or `unknown`. |

### 5.3 ci_attribute_assertions

Purpose: flatten Chest ImaGenome attribute dictionaries into one row per structured assertion.

| Field | Type | Description |
| --- | --- | --- |
| `assertion_id` | string | Unique assertion ID. |
| `image_id` | string | Chest ImaGenome image ID / MIMIC dicom ID. |
| `study_id` | string | MIMIC study ID. |
| `patient_id` | string | MIMIC subject ID. |
| `bbox_name` | string/null | Bound anatomical region. Null if `bbox_name=False`. |
| `anatomy_name` | string/null | Human-readable anatomy name. |
| `anatomy_bound` | bool | Whether this assertion is bound to a valid anatomy. |
| `raw_label` | string | Raw Chest ImaGenome label, e.g. `anatomicalfinding\|no\|pneumothorax`. |
| `category` | string | Parsed category, e.g. `anatomicalfinding`. |
| `polarity` | string | Parsed polarity, e.g. `yes`, `no`. |
| `label_name` | string | Parsed finding name, e.g. `pneumothorax`. |
| `attributes_id` | string/null | Concept ID if available. |
| `phrase_index` | int/null | Index of the supporting phrase. |
| `phrase` | string/null | Report phrase supporting this assertion. |
| `severity_cues` | list[string] | Phrase-level severity cues. |
| `temporal_cues` | list[string] | Phrase-level temporal cues. |
| `texture_cues` | list[string] | Phrase-level texture cues. |
| `comparison_cues` | list[string] | Phrase-level comparison cues. |
| `source_quality` | string | `silver`, `gold`, `manual_checked`, etc. |

### 5.4 radgraph_assertions

Purpose: normalize RadGraph entities and relations into report-side clinical assertions.

| Field | Type | Description |
| --- | --- | --- |
| `assertion_id` | string | Unique assertion ID. |
| `study_id` | string | Linked MIMIC study ID. |
| `patient_id` | string/null | Linked patient ID if available. |
| `entity_id` | string | RadGraph entity ID. |
| `entity_text` | string | Raw entity span text. |
| `entity_type` | string | `ANAT`, `OBS-DP`, `OBS-U`, or `OBS-DA`. |
| `polarity` | string/null | `definitely_present`, `uncertain`, `definitely_absent`, or null. |
| `finding` | string/null | Normalized finding name if entity is an observation. |
| `anatomy` | string/null | Linked anatomy if available. |
| `relations` | list[dict] | `located_at`, `modify`, or `suggestive_of` relations. |
| `source_quality` | string | `silver`, `manual_checked`, etc. |

## Model-ready Item Schema

Model-ready items are constructed from intermediate tables according to `granularity-mapping.md`.

Intermediate tables are not directly used as model input. They are normalized evidence tables produced after parsing raw datasets. Model-ready items are the actual image-question-answer records used for model inference and evaluation.

Each item corresponds to one image-question-answer instance.

Recommended file path:

```text
data/processed/items/{granularity}_{hallucination_probe}_items.jsonl
```

Example files:

```text
data/processed/items/g1_h1_items.jsonl
data/processed/items/g1_h2_items.jsonl
data/processed/items/g2_h1_items.jsonl
data/processed/items/g2_h2_items.jsonl
data/processed/items/g3a_items.jsonl
data/processed/items/g3b_items.jsonl
data/processed/items/g4_proxy_items.jsonl
```

### Required fields

| Field | Type | Description |
| --- | --- | --- |
| `item_id` | string | Unique ID of the constructed item. |
| `source_dataset` | string | Source dataset or dataset combination, such as `ChestImaGenome`, `RadGraph`, or `ChestImaGenome+RadGraph`. |
| `patient_id` | string/null | Patient identifier. Used for linking and leakage control; not exposed to the model. |
| `study_id` | string/null | Study/report identifier. Used to link images with reports and report-derived labels. |
| `image_id` | string/null | Single image identifier, usually equivalent to `dicom_id`. |
| `image_path` | string/null | Local path to the MIMIC-CXR-JPG image used as model input. |
| `granularity` | string | Clinical-semantic granularity level. |
| `question_type` | string | Type of question or experimental probe. |
| `hallucination_probe` | string/null | `H1`, `H2`, or null for non-probe items. |
| `question` | string | Model-facing question. |
| `answer_label` | string/null | Constructed ground-truth answer. |
| `target_finding` | string/null | Target radiological finding, such as `pneumothorax`. |
| `target_anatomy` | string/null | Target anatomical region, such as `right lung`. Null for pure G1 items. |
| `bbox` | object/null | Anatomical-region bbox if applicable. |
| `source_assertions` | list[dict] | Concrete evidence used to construct this item. |
| `evidence_sources` | list[string] | Evidence source types supporting this item. |
| `source_quality` | string | Evidence quality label, such as `silver`, `silver_consistent`, or `manual_checked`. |
| `valid_for_clean_qa` | bool | Whether the item can be used for clean direct QA. |
| `valid_for_false_premise` | bool | Whether the item can be used for false-premise probing. |
| `valid_for_roi_mask` | bool | Whether the item can be used for ROI-mask experiments. |
| `valid_for_roi_only` | bool | Whether the item can be used for ROI-only experiments. |
| `valid_for_training` | bool | Whether the item can be used for SFT/DPO data construction. |
| `exclusion_flag` | string/null | Null if included; otherwise records the exclusion reason. |

```text
clean_direct_qa
multiple_choice_localization
bbox_grounding_probe
false_premise_qa
roi_mask_probe
roi_only_probe
blank_or_shuffled_image_probe
paraphrase_probe
knowledge_only_rewrite
structured_summary
```

### Allowed `answer_label` values

For closed QA:

```text
Yes
No
Uncertain
Supported
Unsupported
A
B
C
D
E
```

For G4 structured synthesis proxy, `answer_label` can be `null`, and the item should use `reference_claims`.

### Example: G2 model-ready item

```json
{
  "item_id": "ci_g2_000001",
  "source_dataset": "ChestImaGenome",
  "patient_id": "10000032",
  "study_id": "50414267",
  "image_id": "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014",
  "image_path": "data/raw/mimic-cxr-jpg/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
  "granularity": "G2_anatomical_localization",
  "question_type": "clean_direct_qa",
  "question": "Is there evidence of pneumothorax in the right lung?",
  "answer_label": "No",
  "target_finding": "pneumothorax",
  "target_anatomy": "right lung",
  "bbox": {
    "bbox_name": "right lung",
    "original_x1": 395,
    "original_y1": 532,
    "original_x2": 1255,
    "original_y2": 1268,
    "coordinate_space": "original_image"
  },
  "source_assertions": [
    {
      "source": "ChestImaGenome",
      "raw_label": "anatomicalfinding|no|pneumothorax",
      "category": "anatomicalfinding",
      "polarity": "no",
      "label_name": "pneumothorax",
      "bbox_name": "right lung",
      "phrases": [
        "Right lung is clear without pneumothorax.",
        "No pneumothorax identified."
      ]
    }
  ],
  "evidence_sources": [
    "E1_bbox",
    "E2_anatomy_finding_pair",
    "E3_report_phrase"
  ],
  "source_quality": "silver",
  "valid_for_clean_qa": true,
  "valid_for_false_premise": true,
  "valid_for_roi_mask": true,
  "valid_for_roi_only": true,
  "valid_for_training": true,
  "exclusion_flag": null
}
```

---

## Model Output Schema

## Prompt Layer Schema

Prompts are split into two JSONL layers so true model inputs do not contain
ground-truth labels.

Model input file:

```text
data/processed/prompts/{split}_model_inputs.jsonl
```

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `item_id` | string | Item ID used to join back to metadata. |
| `image_path` | string | Local image path used by the model runner. |
| `prompt_template_id` | string | Prompt template ID. |
| `prompt` | string | Full text prompt sent to the model. |

Evaluation metadata file:

```text
data/processed/prompts/{split}_eval_metadata.jsonl
```

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `item_id` | string | Item ID used to join to model output. |
| `answer_label` | string/null | Ground-truth answer for scoring. |
| `granularity` | string | Granularity level. |
| `question_type` | string | Question type or experimental probe. |
| `hallucination_probe` | string/null | `H1`, `H2`, or null. |
| `target_finding` | string/null | Target finding. |
| `target_anatomy` | string/null | Target anatomy. |
| `evidence_sources` | list[string] | Evidence source types. |

Model outputs should be saved separately from model-ready items.

The purpose of this schema is to preserve the raw model response, the parsed answer, and the generation configuration. This makes debugging, re-parsing, and manual review possible.

Recommended file path:

```text
outputs/raw_responses/{model_name}_{split}.jsonl
```

### Required field

| Field                | Type        | Description                                           |
| -------------------- | ----------- | ----------------------------------------------------- |
| `item_id`            | string      | ID of the evaluated item.                             |
| `model_name`         | string      | Name of the evaluated model.                          |
| `model_version`      | string/null | Model version, checkpoint, or commit ID if available. |
| `image_path`         | string/null | Image path used for inference.                        |
| `prompt_template_id` | string/null | Prompt template ID.                                   |
| `prompt`             | string      | Actual prompt sent to the model.                      |
| `raw_response`       | string      | Original unmodified model output.                     |
| `parsed_answer`      | string/null | Parsed answer, such as `Yes`, `No`, `Supported`, `Unsupported`, or `Uncertain`. |
| `parse_status`       | string      | Status of answer parsing.                             |
| `generation_config`  | object      | Decoding configuration.                               |
| `runtime`            | object      | Runtime metadata.                                     |

### Allowed `parse_status` values

```text
success
invalid_format
multiple_answers
empty_response
requires_manual_review
```

### Example

```json
{
  "item_id": "ci_g2_000001",
  "model_name": "qwen3-vl-4b-instruct",
  "model_version": "model-card-or-commit",
  "image_path": "data/raw/mimic-cxr-jpg/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
  "prompt_template_id": "yes_no_v1",
  "prompt": "You are given a chest X-ray. Answer only one of: Yes, No, or Uncertain. Question: Is there evidence of pneumothorax in the right lung?",
  "raw_response": "Yes, there is evidence of pneumothorax in the right lung.",
  "parsed_answer": "Yes",
  "parse_status": "success",
  "generation_config": {
    "temperature": 0.0,
    "max_new_tokens": 64,
    "seed": 42
  },
  "runtime": {
    "timestamp": "YYYY-MM-DDTHH:MM:SS",
    "device": "cuda:0"
  }
}
```

---

## Scored Output Schema

Scored outputs are produced after comparing model outputs with the constructed answer labels and evidence sources.

The purpose of this schema is to store correctness, hallucination type, evidence relation, and other error categories.

Recommended file path:

```text
outputs/scored/{model_name}_{split}_scored.jsonl
```

### Required Fields

| Field                    | Type        | Description                                          |
| ------------------------ | ----------- | ---------------------------------------------------- |
| `item_id`                | string      | ID of the evaluated item.                            |
| `model_name`             | string      | Name of the evaluated model.                         |
| `granularity`            | string      | Granularity level of the item.                       |
| `question_type`          | string      | Question type or experimental probe.                 |
| `answer_label`           | string/null | Constructed ground-truth answer.                     |
| `parsed_answer`          | string/null | Parsed model answer.                                 |
| `score`                  | string      | Final evaluation score.                              |
| `is_correct`             | bool/null   | Whether the parsed answer is correct.                |
| `hallucination_type`     | string/null | H1, H2, or null.                                     |
| `h1_error_direction`     | string/null | `false_positive` for No->Yes, `false_negative` for Yes->No, or null. |
| `evidence_relation`      | string      | Relation between model claim and available evidence. |
| `error_category`         | string/null | Non-hallucination reliability issue if applicable.   |
| `requires_manual_review` | bool        | Whether manual review is required.                   |
| `notes`                  | string/null | Optional notes for error analysis.                   |

### Allowed `score` values

```text
correct
incorrect_non_hallucination
H1_evidence_contradicted
H2_evidence_unsupported
uncertain_or_abstention
invalid_response
requires_manual_review
```

### Allowed `hallucination_type` values

```text
null
H1
H2
```

### Allowed `h1_error_direction` values

```text
null
false_positive
false_negative
```

### Allowed `error_category` values

```text
null
omission
medical_knowledge_error
cross_granularity_inconsistency
uncertain_case
```

### Allowed `evidence_relation` values

```text
S2_directly_supported
S1_weakly_supported
S0_unsupported
S-1_contradicted
U_uncertain
```

### Basic scoring rule for closed Yes/No QA

| Ground truth     | Parsed answer | Default score                                |
| ---------------- | ------------- | -------------------------------------------- |
| `Yes`            | `Yes`         | `correct`                                    |
| `No`             | `No`          | `correct`                                    |
| `Yes`            | `No`          | `H1_evidence_contradicted`                   |
| `No`             | `Yes`         | `H1_evidence_contradicted`                   |
| `Uncertain`      | `Yes` or `No` | `requires_manual_review` or `uncertain_case` |
| `Yes` or `No`    | `Uncertain`   | `uncertain_or_abstention`                    |
| invalid response | any           | `invalid_response`                           |

### Basic scoring rule for claim-support QA

| Ground truth  | Parsed answer | Default score                 |
| ------------- | ------------- | ----------------------------- |
| `Supported`   | `Supported`   | `correct`                     |
| `Unsupported` | `Unsupported` | `correct`                     |
| `Unsupported` | `Supported`   | `H2_evidence_unsupported`     |
| `Supported`   | `Unsupported` | `incorrect_non_hallucination` |
| `Supported` or `Unsupported` | `Uncertain` | `uncertain_or_abstention` |
| invalid response | any        | `invalid_response`            |

### H2 pilot sampling

H2 item builders support configurable downsampling of `Supported` and
`Unsupported` claims. The first pilot default targets 80% `Unsupported` and
20% `Supported`, but this is an experiment setting rather than an optimal test
set definition. Scripts should expose the target fraction and seed so later
sensitivity analyses can rerun with different ratios.

### Examples

```json
{
  "item_id": "ci_g2_000001",
  "model_name": "qwen3-vl-4b-instruct",
  "granularity": "G2_anatomical_localization",
  "question_type": "clean_direct_qa",
  "answer_label": "No",
  "parsed_answer": "Yes",
  "score": "H1_evidence_contradicted",
  "is_correct": false,
  "hallucination_type": "H1",
  "evidence_relation": "S-1_contradicted",
  "error_category": null,
  "requires_manual_review": false,
  "notes": null
}
```

---

## Global Validation Rules

The following rules must be enforced by data parsers, item builders, inference scripts, and scoring scripts.

1. Missing evidence must not be converted into negative evidence.
2. `Unknown` and `Conflict` must be treated as separate states.
3. `bbox_name=False` may support G1 if an explicit finding assertion exists, but must not support G2.
4. G1 requires a finding-level positive, negative, or uncertain assertion.
5. G2 requires valid anatomy-finding binding.
6. G2 bbox refers to anatomical-region localization, not lesion segmentation.
7. G3a requires a target anatomy and a plausible distractor anatomy.
8. G3a must not be reduced to a simple G2 anatomy-specific assertion.
9. G3b requires positive finding assertion, non-empty modifier cue or RadGraph `modify` relation, and phrase-level validation.
10. Negated findings must not be used for modifier characterization.
11. G4 is a structured synthesis proxy, not definitive clinical diagnosis.
12. Items with conflicting evidence should be excluded from the first pilot or marked as `requires_manual_review`.
13. Raw restricted medical images, reports, and patient-linked outputs must not be committed to Git.
14. Public fixtures must use toy examples or fully de-identified synthetic records.
15. Every model-ready item must include `source_assertions` and `evidence_sources`.
16. Every scored output must preserve `item_id`, `granularity`, `question_type`, `answer_label`, `parsed_answer`, and `score`.

---

## Training Data Schema
