# Study 2

## G1

The 'attributes' in the lists follow the pattern of < categoryID | relation | label_name >
Example:
'attributes': [['anatomicalfinding|no|lung opacity',
    'anatomicalfinding|no|pneumothorax',
    'nlp|yes|normal'],
    ['anatomicalfinding|no|pneumothorax']],
    - 'attributes_ids': [['CL556823', 'C1963215;;C0032326', 'C1550457'],
   ['C1963215;;C0032326']],

### Construct rules

For every anatomicalfindings mentioned in the 'attributes' of this image ,we construct 2 questions.
Question 1:
You are given a chest X-ray.
Evaluate the following medical claim based only on the visible radiographic evidence.
Claim: There is evidence of < label_name > in this chest X-ray.
Answer only one of:
A. Supported
B. Contradicted
C. Not enough evidence
Question 2:
You are given a chest X-ray.
Evaluate the following medical claim based only on the visible radiographic evidence.
Claim: There isn't evidence of < label_name > in this chest X-ray.
Answer only one of:
A. Supported
B. Contradicted
C. Not enough evidence
If < relation > is yes, the right answer for Question 1 is A ,and for Question 2 is B.
If < relation > is no, the right anwser for Question 1 is B ,and for Question 2 is A.

For unmentioned findings, use a fixed Chest ImaGenome anatomical finding vocabulary ,we also construct 2 questions as above.
Missing evidence must be labeled as Not enough evidence, not Contradicted.
As we can't judge by yes or no ,so the right answer for Question 1 and 2 is C.

## G2

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

We just replace "in this chest X-ray" by the 'bbox_name' ,and construct 2 questions too.
The evaluation rules is the same.

## Statistical indicators

We count the accuracy of Question 1 and Question 2 ,for G1 ,G2 ,and both.
We count the answer disribution for each question.

## All anatomicalfindings

The all anatomical findings in the ChestImaGenome.

lung opacity, airspace opacity, consolidation, infiltration, atelectasis, linear/patchy atelectasis, lobar/segmental collapse, pulmonary edema/hazy opacity, vascular congestion, vascular redistribution, increased reticular markings/ild pattern, pleural effusion, costophrenic angle blunting, pleural/parenchymal scarring, bronchiectasis, enlarged cardiac silhouette, mediastinal displacement, mediastinal widening, enlarged hilum, tortuous aorta, vascular calcification, pneumomediastinum, pneumothorax, hydropneumothorax, lung lesion, mass/nodule (not otherwise specified), multiple masses/nodules, calcified nodule, superior mediastinal mass/enlargement, rib fracture, clavicle fracture, spinal fracture, hyperaeration, cyst/bullae, elevated hemidiaphragm, diaphragmatic eventration (benign), subdiaphragmatic air, subcutaneous air, hernia, scoliosis, spinal degenerative changes, shoulder osteoarthritis, bone lesion
