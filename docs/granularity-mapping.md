# Granularity Mapping for Chest ImaGenome
This is the detailed definition of granularity in Chest Imagenome, corresponding dataset fields, evidence sources, sample construction rules, exclusion rules, and example items.
## G1
Finding Existence refers to image-level existence of a radiological finding or abnormality.
### dataset mapping
Scene Graph JSON - attributes field
- example: ‘attributes’: [[‘anatomicalfinding|no|lung opacity’, ‘anatomicalfinding|no|pneumothorax’, ‘nlp|yes|normal’], ‘anatomicalfinding|no|pneumothorax’]
### question template
Is there evidence of {finding} in this chest X-ray?
### Construction rule
- Yes: at least one relevant anatomical region has `{category}|yes|{finding}`.
- No: explicit negative evidence exists for `{finding}` and no positive evidence conflicts with it.
- Unknown: finding is not mentioned or evidence is incomplete.
### example: 
#### Positive G1
Source assertion:
`right lower lung zone: anatomicalfinding|yes|lung opacity`

Constructed item:
- Granularity: G1_finding_existence
- Question: Is there evidence of lung opacity in this chest X-ray?
- Answer: Yes
- Evidence: At least one anatomical region has `lung opacity=yes`.
#### Negative G1
Source assertion:
`right lung: anatomicalfinding|no|pneumothorax`

Constructed item:
- Granularity: G1_finding_existence
- Question: Is there evidence of pneumothorax in this chest X-ray?
- Answer: No
- Evidence: Explicit negative pneumothorax assertion and no positive pneumothorax assertion.

## G2


## G3
### property
1. 

## G4 
