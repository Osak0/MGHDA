# Study 3 Multi-select Construction v2

## Research question

Study 3 tests whether a medical multimodal model can distinguish findings with
explicit positive (`yes`) evidence from findings with explicit negative (`no`)
evidence. It follows the Study 2 observation that reverse/negative claim
construction is difficult and may trigger response shortcuts.

Missing, Unknown, Conflict, and unmentioned findings are excluded. Missing
labels are never converted to `no`.

## Identity and evidence scope

```text
experiment_id = study3_multiselect_v2
question_type = anatomicalfinding_multiselect_v2
```

- G1 uses explicit image-level `anatomicalfinding` yes/no assertions.
- G2 additionally requires an explicit anatomy binding and a complete bbox.
- Chest ImaGenome bboxes are anatomical regions, not lesion segmentations.
- Every option is traceable through private `source_assertions`.

## Four-question design

One option set produces four independent records:

```text
state + present
state + absent
evidence + present
evidence + absent
```

Metadata includes:

```text
prompt_framing = state | evidence
query_relation = present | absent
```

The item ID is derived from `option_set_id`, `prompt_framing`, and
`query_relation`. The four records share the same image, option set, and option
order. State/evidence records for the same relation share the same gold set.
Present and absent gold sets are complements because only explicit yes/no
options are admitted.

State prompts retain the v1 wording:

```text
Considering only the listed findings, select all findings that are
present/absent in the scope.
```

Evidence-present:

```text
Considering only the listed findings, select all findings for which the
radiographic evidence supports presence in the scope.
```

Evidence-absent:

```text
Considering only the listed findings, select all findings for which the
radiographic evidence explicitly supports absence in the scope.

Evidence supporting absence means that the finding is ruled out by the
radiographic evidence. It does not mean that the finding is merely unmentioned.
```

All prompts end with the same strict response format: comma-separated option
letters, or `NONE`.

## Sampling and fixed scale

Seed is 42.

Per granularity:

- 250 natural anchors × 4 questions = 1,000 records.
- 50 controlled anchors × K=2/3/4/5 × 4 questions = 800 records.
- Total = 1,800 records.

G1+G2 total = 3,600 records. Controlled options are nested so K=2 is a prefix
of K=3, and so on. Candidate shortages are reported rather than filled with
unmentioned findings.

The fixed smoke subset has 40 records: one for every
`granularity × framing × relation × {natural,K2,K3,K4,K5}` cell.

## Private paths

```text
processed/study3/v2/items/
processed/study3/v2/prompts/
processed/study3/v2/smoke/
outputs/study3/v2/
```

Model inputs contain only item ID, experiment ID, image path, template ID, and
prompt. Gold, polarity, source assertions, and evidence metadata remain in
eval-metadata JSONL.

## Metrics

Report overall and stratified exact-set accuracy, Hamming accuracy, Jaccard,
set precision/recall/F1, micro metrics, NONE accuracy, invalid rate, option
position effects, and:

- by `prompt_framing`;
- by `query_relation`;
- framing × relation;
- state/evidence paired agreement;
- complement consistency separately within each framing;
- controlled-K paired bootstrap separately within each framing.

Infrastructure failures are validation failures. Invalid model responses are
research outcomes and are reported separately.
