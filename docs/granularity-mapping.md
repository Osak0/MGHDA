# Granularity Mapping

## Chest ImaGenome
This is the detailed definition of granularity in Chest Imagenome, corresponding dataset fields, evidence sources, sample construction rules, exclusion rules, and example items.
---
### G1
Finding Existence refers to image-level existence of a radiological finding or abnormality.
#### dataset mapping
Scene Graph JSON - attributes field
- example: ‘attributes’: [‘anatomicalfinding|no|lung opacity’, ‘anatomicalfinding|no|pneumothorax’]
#### question template
Is there evidence of {finding} in this chest X-ray?
#### Construction rule
- Yes: at least one relevant anatomical region has `{category}|yes|{finding}`.
- No: no positive evidence conflicts with it, has `{category}|no|{finding}`.
- Unknown: finding is not mentioned or evidence is incomplete.
#### example: 
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
### G2
Anatomical Localization refers to anatomy-specific existence of a radiological finding.
#### dataset mapping
Scene Graph JSON - objects field
example:
    - ‘bbox_name’: ‘right upper lung zone’,
    - 'original_x1': 395,
    - 'original_y1': 532,
    - 'original_x2': 1255,
    - 'original_y2': 1268,
    - 'original_width': 860,
    - 'original_height': 736
#### question template
- Is there ... in the ... ?
- Where is the {findings} located? A... B... C... D... E.no evidence of the {...}
- as model could complete bbox grounding task: extension question for (x1, x2, y1, y2)
#### construction rule
For G2-main yes/no QA:
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
#### example:
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
### G3
#### relation
##### finding-anatomy binding
'attributes': 'anatomicalfinding|no|lung opacity'
‘bbox_name’: ‘right upper lung zone’
##### question template


### G4
G4 refers to image-level or report-level synthesis based on multiple localized findings. In Chest ImaGenome, G4 is treated only as a structured synthesis proxy, not definitive clinical diagnosis.
missing
---
## 