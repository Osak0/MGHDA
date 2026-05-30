# Granularity Mapping

## MIMIC-CXR

the source dataset
Based on overlapping MIMIC-CXR studies linked with Chest ImaGenome and RadGraph, we construct a derived multi-granularity experimental subset covering operationalizable parts of G1–G4.

---

## Chest ImaGenome

This is the detailed definition of granularity in Chest Imagenome, corresponding dataset fields, evidence sources, sample construction rules, exclusion rules, and example items.

---

### G1 (Chest ImaGenome)

Finding Existence refers to image-level existence of a radiological finding or abnormality.

#### dataset mapping (G1)

Scene Graph JSON - attribute dictionary

example
    - 'attributes': [['anatomicalfinding|no|lung opacity',
    'anatomicalfinding|no|pneumothorax',
    'nlp|yes|normal'],
    ['anatomicalfinding|no|pneumothorax']],
    - 'attributes_ids': [['CL556823', 'C1963215;;C0032326', 'C1550457'],
   ['C1963215;;C0032326']],

#### question template (G1)

addition: For G1, we use the `attributes` lists from Chest ImaGenome attribute dictionaries, regardless of whether `bbox_name` is valid or False, because G1 only requires image-level finding existence or absence and does not require anatomy-specific localization.

Is there evidence of {finding} in this chest X-ray?

#### Construction rule (G1)

- Yes: at least one relevant anatomical region has `{category}|yes|{finding}`.
- No: no positive evidence conflicts with it, has `{category}|no|{finding}`.
- Unknown: `{finding}` is not mentioned or evidence is incomplete.
- Conflict: both positive and negative assertions exist for the same finding; exclude from the first pilot or send to manual review.

#### example (G1)

##### Positive G1

Source assertion:
`right lower lung zone: anatomicalfinding|yes|lung opacity`

Constructed item:
    - Granularity: G1_finding_existence
    - Question: Is there evidence of lung opacity in this chest X-ray?
    - Answer: Yes
    - Evidence: At least one anatomical region has `lung opacity=yes`.

##### Negative G1

Source assertion:
`right lung: anatomicalfinding|no|pneumothorax`

Constructed item:
    - Granularity: G1_finding_existence
    - Question: Is there evidence of pneumothorax in this chest X-ray?
    - Answer: No
    - Evidence: Explicit negative pneumothorax assertion and no positive pneumothorax assertion.
  
---

### G2 (Chest ImaGenome)

Anatomical Localization refers to anatomy-specific existence of a radiological finding.

#### dataset mapping (G2)

Scene Graph JSON
objects dictionary
example (bbox metadata):
    - ‘bbox_name’: ‘right upper lung zone’,
    - 'original_x1': 395,
    - 'original_y1': 532,
    - 'original_x2': 1255,
    - 'original_y2': 1268,
    - 'original_width': 860,
    - 'original_height': 736

attribute dictionary
example (attributes list):
    - 'right lung': True,
    'bbox_name': 'right lung',
    'synsets': ['C0225706'],
    'name': 'Right lung',
    - 'attributes': [['anatomicalfinding|no|lung opacity',
    'anatomicalfinding|no|pneumothorax',
    'nlp|yes|normal'],
    ['anatomicalfinding|no|pneumothorax']],
    - 'attributes_ids': [['CL556823', 'C1963215;;C0032326', 'C1550457'],
    ['C1963215;;C0032326']],

#### question template (G2)

- Is there {attributes{findings}} in the {bbox_name} ?
- Where is the {findings} located? A... B... C... D... E.no evidence of the {...}
- as model could complete bbox grounding task: extension question for (x1, x2, y1, y2)

#### construction rule (G2)

addition:the key of the first line in attribute dictionary that refers to anatomy name, such as'right lung', need to be True, because False means anatomical location may not always be described or implied in the report, but in G2, we need to focus on the anatomical location level, so the anatomy should be bound to the attribute dictionary.

For G2-main yes/no QA
    - Yes: the target anatomy has `{category}|yes|{finding}`.
    - No: the target anatomy has explicit `{category}|no|{finding}` and no conflicting positive assertion for the same anatomy-finding pair.
    - Unknown: the target anatomy is absent, the finding is not mentioned for that anatomy, or the evidence is incomplete.
    - Conflict: the same anatomy-finding pair has both positive and negative evidence; exclude from pilot or mark for manual review.

For G2-choice:
    - Construct only when one or more positive anatomy-finding pairs exist.
    - Candidate choices should be anatomically plausible and preferably symmetric or nearby regions.
    - Include `no evidence of this finding` only when image-level G1 label is No.

For G2-box:
    - Use only as an optional grounding probe.
    - Use `original_x1`, `original_y1`, `original_x2`, `original_y2` as the reference bbox when evaluating on original MIMIC-CXR-JPG images.
    - Treat this as anatomical-region localization, not lesion segmentation.

#### examples (G2 — anatomical localization)

##### Positive G2 example

Source assertion:
`right upper lung zone: anatomicalfinding|yes|lung opacity`

Object evidence:
`bbox_name = right upper lung zone`
`original_x1 = 395, original_y1 = 532, original_x2 = 1255, original_y2 = 1268`

Constructed item:
    - Granularity: G2_anatomical_localization
    - Question: Is there evidence of lung opacity in the right upper lung zone?
    - Answer: Yes
    - Evidence: The right upper lung zone has `anatomicalfinding|yes|lung opacity`.
    - Bbox: anatomical region bbox, not lesion segmentation.

##### Negative G2 example

Source assertion:
`right lung: anatomicalfinding|no|pneumothorax`

Constructed item:
    - Granularity: G2_anatomical_localization
    - Question: Is there evidence of pneumothorax in the right lung?
    - Answer: No
    - Evidence: The right lung has `anatomicalfinding|no|pneumothorax`, and no positive pneumothorax assertion conflicts with it.

##### Optional G2-box example

Question:
Draw a bounding box around the right upper lung zone. Return JSON only:
{"bbox": [x1, y1, x2, y2]}

Reference:
`[original_x1, original_y1, original_x2, original_y2]`

Evaluation:
    - IoU with anatomical bbox;
    - or center point inside reference bbox;
    - not used as main G2 hallucination metric.

---

### G3 (Chest ImaGenome)

#### G3a: Contrastive finding-anatomy binding

G3a tests whether a model can correctly bind a finding to the intended anatomical region while rejecting an anatomically plausible but incorrect region, is not a simple anatomy-specific assertion, it is a contrastive relation-level probe built from two or more candidate anatomy-finding bindings.

##### Dataset mapping (G3a)

- attribute dictionary: `{category}|yes/no|{finding}`
- valid `bbox_name`
- paired or contrastive anatomy candidates
- optional phrases for report-side evidence

##### Question templates (G3a)

1. Is the {finding} located in the {anatomy_A} rather than the {anatomy_B}?
2. The {finding} is located in the {anatomy_B}. Is this statement supported by the chest X-ray?
3. Which anatomical region is more consistent with the {finding}? A. {anatomy_A} B. {anatomy_B} C. no evidence of this finding

##### Construction rule (G3a)

- Positive binding: target anatomy has `{category}|yes|{finding}`.
- Negative binding: distractor anatomy has `{category}|no|{finding}` or is selected as an anatomically plausible but unsupported contrast region.
- Prefer symmetric or nearby distractors, such as left/right lung zones or upper/lower lung zones.
- Exclude cases where both anatomies have positive evidence or where the distractor is ambiguous.

#### G3b: modifier

attribute dictionary:
'comparison_cues': [[], []],
'temporal_cues': [[], []],
'severity_cues': [[], []],
'texture_cues': [[], []],

##### Construction rule(G3b)

addition:These cue fields are aligned at the phrase / sentence level with the outer lists of `phrases` and `attributes`. Because severity and temporal cues are assigned by sentence-level co-occurrence, they are treated as weak/silver evidence and require phrase-level validation.

- Use only positive finding assertions, e.g. `{category}|yes|{finding}`
- The corresponding cue list must be non-empty.
- The supporting phrase must contain both the target finding and the modifier cue.
- Exclude negated findings.
- Exclude cases where multiple findings in the same phrase make modifier binding ambiguous.
- Treat all G3b items as weak/silver evidence unless manually validated.

### G4 — structured synthesis (Chest ImaGenome)

G4 refers to image-level or report-level synthesis based on multiple localized findings. In Chest ImaGenome, G4 is treated only as a structured synthesis proxy, not definitive clinical diagnosis.

---

## RadGraph

RadGraph is used as report-side structured evidence, not direct visual ground truth.

### Schema

Entities:
    - ANAT: Anatomy
    - OBS-DP: Observation Definitely Present
    - OBS-U: Observation Uncertain
    - OBS-DA: Observation Definitely Absent

### G1 (RadGraph)

- OBS-DP -> Yes
- OBS-DA -> No
- OBS-U -> Uncertain, excluded from binary clean QA

### G2 (RadGraph)

- Use `located_at(Observation, Anatomy)` as report-side localization evidence.
- This is E3 report-side localization evidence, not bbox or segmentation.

### G3 (RadGraph)

- `modify(Observation, Observation)` or `modify(Anatomy, Anatomy)` -> candidate modifier evidence.
- `suggestive_of(Observation, Observation)` -> candidate higher-level inference evidence, reserved for G4 proxy or diagnostic synthesis analysis.

## Evidence Source Mapping

| Granularity | Chest ImaGenome Evidence | RadGraph Evidence | Evidence level |
| --- | --- | --- | --- |
| G1 | attribute `{category}\|yes/no\|finding` | OBS-DP / OBS-DA | E2 / E3 |
| G2 | valid `bbox_name` + anatomy-specific attribute | located_at(Observation, Anatomy) | E1 + E2 / E3 |
| G3a | contrastive anatomy-finding binding | located_at contrast | E2 / E3 |
| G3b | severity/temporal/texture/comparison cues + phrases | modify relation | weak E3 |
| G4 | aggregated localized findings | aggregated report graph / suggestive_of | E3 proxy |
