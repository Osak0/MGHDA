# Study 3 Explicit Anatomical-Finding Multi-Select Construction

This document is the authoritative experiment specification for Study 3. Study
3 is independent from the Study 2 claim-verification experiment defined in
`docs/study2-prompt-construction.md`; its code, item IDs, prompt schemas,
generated data, transfer manifests, inference outputs, and metrics must not be
mixed with Study 2.

## 1. Research question and scope

Study 3 asks a model to select a set of explicit anatomical findings:

1. Which listed findings are present?
2. Which listed findings are absent?

The same G1 and G2 taxonomy definitions remain in force:

- `G1_finding_existence`: finding existence across the whole image.
- `G2_anatomical_localization`: finding existence in one named anatomical
  region.

Study 3 uses only explicit Chest ImaGenome
`anatomicalfinding|yes/no|{finding}` assertions. It does not add unmentioned
findings and therefore does not directly evaluate the Study 2
`H2_evidence_unsupported` condition. Its primary purposes are explicit
yes/no-evidence recognition, present/absent wording robustness, and option-count
sensitivity.

Chest ImaGenome attributes are report-derived silver evidence. Study 3 scores
agreement with these structured assertions; it must not be described as direct
pixel-ground-truth diagnostic accuracy. Dataset structure and relation semantics
follow the official [Chest ImaGenome 1.0.0
documentation](https://physionet.org/content/chest-imagenome/1.0.0/).

Option count and order are treated as experimental factors because multiple
choice performance can be sensitive to both [option
ordering](https://aclanthology.org/2024.findings-naacl.130/) and [option
number](https://aclanthology.org/2025.findings-acl.1138/).

## 2. Experiment identity and isolation

All Study 3 records use:

```text
experiment_id = study3_multiselect_v1
question_type = anatomicalfinding_multiselect_v1
prompt_template_id = study3_multiselect_present_v1
                     or study3_multiselect_absent_v1
```

Item IDs start with `study3_g1_ms_` or `study3_g2_ms_`. Anchor and option-set
IDs also start with `study3_`. Study 3 code must reject a different experiment
ID, a non-Study-3 item ID, or an output path beginning with a Study 2-specific
name.

Study 3 generated artifacts live only under:

```text
interim/study3/
processed/study3/items/
processed/study3/prompts/
outputs/study3/audits/
outputs/study3/raw_responses/
outputs/study3/scored/
outputs/study3/figures/
outputs/study3/transfer/
```

Images remain in the shared restricted `files/` tree and are not duplicated.

## 3. Explicit-attribute eligibility

An assertion is eligible only when:

- `category == anatomicalfinding`;
- `polarity` is exactly `yes` or `no`;
- `image_id` and `label_name` are non-empty.

Study 3 never converts a missing attribute into `no`. `Unknown` and `Conflict`
remain separate states and do not enter an option list.

Duplicate assertions for the same scope and finding are collapsed. The private
eval metadata retains all cleaned `source_assertions`; raw phrases are not
copied.

### 3.1 G1 aggregation

The G1 anchor is one image. Assertions are grouped by:

```text
patient_id + study_id + image_id + label_name
```

- only `yes` -> the finding is present;
- only `no` -> the finding is absent;
- both `yes` and `no` -> exclude that finding as Conflict.

An image is eligible only after at least two non-conflicting findings remain.

### 3.2 G2 aggregation

The G2 anchor is one image and one named anatomy:

```text
patient_id + study_id + image_id + bbox_name
```

An assertion must have `anatomy_bound=True`, a real `bbox_name`, and a matching
Chest ImaGenome object with a complete normalized or original coordinate set.
`bbox_name=False` and incomplete/missing object boxes cannot support G2.

Assertions are grouped by the anchor plus `label_name`; yes/no conflicts are
excluded. The bbox denotes an anatomical region, not a lesion segmentation.
At least two non-conflicting findings must remain.

## 4. Local-image gate

Candidate anchors are linked through the official MIMIC-CXR-JPG metadata and
split tables. An anchor is eligible for sampling only when:

- its subject/study/dicom identifiers match metadata;
- its portable path can be constructed under `files/`;
- the referenced image already exists locally.

Study 3 does not download images. Candidate audits emit aggregate counts only:
K bins, answer composition, local-image exclusions, and controlled-K
availability. They must not print patient, study, image, report, phrase, or path
values.

## 5. Natural option sets

The reference run requests 500 natural anchors per granularity:

- G1: 500 images;
- G2: 500 image+bbox anchors.

Each natural option set contains every eligible explicit finding for that
anchor. Sampling targets four natural-K strata:

```text
K=2, K=3, K=4, K>=5
```

For 500 anchors the nominal quota is 125 per stratum. Within each K stratum,
sampling round-robins across:

- `all_yes`;
- `all_no`;
- `mixed`.

Sampling is deterministic with seed 42. When a K stratum is short, remaining
eligible anchors from other strata fill the total. No finding or anchor is
invented or duplicated. A remaining overall shortage is recorded explicitly.

Every natural option set creates a present question and an absent question.
The target is therefore 1,000 natural items per granularity.

## 6. Controlled nested option sets

From the selected natural anchors, choose up to 100 anchors with at least five
explicit findings. The controlled ranking again round-robins across answer
composition and is deterministic.

For one controlled anchor, build one stable option permutation and its nested
prefixes:

```text
K=2 -> first 2 findings
K=3 -> first 3 findings
K=4 -> first 4 findings
K=5 -> first 5 findings
```

The present and absent questions for one option set use exactly the same
options and order. They are independent inference records and must not share a
conversation context.

With 100 controlled anchors, four K values, and two query relations, the target
is 800 controlled items per granularity. If fewer than 100 K>=5 anchors exist,
use all available anchors and report the shortage; never pad with unmentioned
findings.

Reference target:

| Granularity | Natural | Controlled | Total |
|---|---:|---:|---:|
| G1 | 1,000 | 800 | 1,800 |
| G2 | 1,000 | 800 | 1,800 |
| Combined | 2,000 | 1,600 | 3,600 |

## 7. Question and gold construction

The strict English template is:

```text
This is a multiple-select question.
Considering only the listed findings, select all findings that are absent
in the right lung.

A. lung opacity
B. pneumothorax

Reply only with comma-separated option letters, or NONE if no option applies.
```

For G1, the scope is `in this chest X-ray`. For G2, the scope is
`in the {bbox_name}`.

- present question gold = option IDs whose explicit polarity is `yes`;
- absent question gold = option IDs whose explicit polarity is `no`;
- an empty gold set is represented by `NONE`;
- an all-selected set is represented by every option ID.

For:

```text
anatomicalfinding|no|lung opacity
anatomicalfinding|no|pneumothorax
```

the present answer is `NONE` and the absent answer is `A,B`.

## 8. Prompt and metadata layers

Model-input JSONL contains exactly:

```text
item_id
experiment_id
image_path
prompt_template_id
prompt
```

Private eval-metadata JSONL contains:

```text
item_id
experiment_id
granularity
question_type
query_relation
variant
controlled_k
anchor_id
option_set_id
option_count
natural_option_count
answer_composition
options
gold_selected_options
target_anatomy
bbox
source_assertions
evidence_sources
source_quality
```

Gold sets, source polarities, and evidence assertions must never enter model
inputs. Both layers are restricted artifacts and join only by `item_id`.

## 9. Answer parsing

The parser is case-insensitive and accepts:

- `A,C`;
- `A C`;
- `A and C`;
- bracketed forms such as `[A,C]`;
- `NONE`.

It rejects:

- empty output;
- `NONE` combined with option IDs;
- duplicate IDs;
- IDs outside the current option set;
- additional prose or an ambiguous format.

Invalid model responses are separate from infrastructure failures.

## 10. Metrics and interpretation

Primary metrics:

- option-level Hamming accuracy;
- micro precision, recall, and F1;
- mean set F1;
- Jaccard.

Secondary metrics:

- exact-set accuracy;
- `NONE` accuracy;
- invalid-response rate;
- present/absent complement consistency;
- option-position accuracy and selection rate.

Report every metric by:

- G1/G2;
- present/absent;
- natural/controlled;
- option count K;
- all-yes/all-no/mixed;
- option position.

For controlled items, average the paired present/absent Hamming score for each
anchor and K. Compare K=2 with K=3/4/5 by anchor-level paired bootstrap with
10,000 resamples and a 95% percentile interval.

Exact-set accuracy is never the only cross-K metric. If exact-set accuracy
falls while option-level Hamming accuracy is stable, interpret the decline as
largely mechanical. A decline in Hamming accuracy provides stronger evidence
that additional options burden the model.

G1/G2 differences are descriptive unless the populations are explicitly
matched. Any result remains conditional on report-derived silver evidence.

## 11. Transfer and reproducibility

Study 3 uses:

```text
outputs/study3/transfer/study3_files_from.txt
outputs/study3/transfer/study3_files.sha256
outputs/study3/transfer/study3_transfer_summary.json
```

The summary records experiment ID, schema version, Git commit, record counts,
G1/G2-natural/controlled-present/absent counts, template IDs, unique images,
payload files, and bytes. The transfer directory must be empty before
generation to prevent stale-file mixing.

The GPU host must check out the exact recorded Git commit. Preflight verifies
the experiment identity, checksums, commit, prompt/metadata ID equality, image
existence, and environment/model readiness before inference.

Generation uses deterministic decoding by default (`temperature=0`), seed 42,
per-item checkpoints, and resume. Study 3 run names start with
`medgemma_study3_multiselect_v1`.

## 12. Restricted-data rules

Real Chest ImaGenome/MIMIC inputs, images, prompt JSONL, eval metadata, raw
responses, checkpoints, and scored row-level outputs remain restricted. They
must not be committed, uploaded to public services, printed in logs, or copied
into documentation.

Only aggregate summaries containing counts and metrics are safe to share after
confirming that they contain no identifiers, paths, report text, or raw
phrases. Repository tests use synthetic fixtures only.
