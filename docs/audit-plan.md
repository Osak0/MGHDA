# Audit Plan

## 1. Purpose

Define the pre-pilot audit work required before constructing model-ready
multi-granularity items from MIMIC-CXR / MIMIC-CXR-JPG, Chest ImaGenome, and
RadGraph. The audit should verify schema assumptions, linking coverage, evidence
quality, and whether the first pilot can be built without converting missing or
ambiguous evidence into negative labels.

## 2. Chest ImaGenome Audit Checklist

- Count Chest ImaGenome scene graph files.
- Count available object records and anatomical bbox records.
- Count top anatomical finding labels.
- Count `yes`, `no`, `uncertain`, and `conflict` assertion states.
- Count `bbox_name=False` records.
- Count non-empty `severity_cues`, `temporal_cues`, `texture_cues`, and
  `comparison_cues`.
- Check category distribution and finding distribution.
- Check phrase coverage for structured labels.
- Verify that `attribute` is treated as a dataset field, not as G3 clinical
  attribute evidence by itself.

## 3. RadGraph Audit Checklist

- Count RadGraph studies.
- Count entity types: `ANAT`, `OBS-DP`, `OBS-U`, and `OBS-DA`.
- Count relation types including `located_at`, `modify`, and `suggestive_of`.
- Count top observations and top anatomies.
- Count RadGraph study overlap with MIMIC-CXR.
- Count RadGraph study overlap with Chest ImaGenome-linked studies.

## 4. MIMIC-CXR-JPG Linking Checklist

- Count linked `image_id`, `study_id`, and `image_path` records.
- Check one-to-many relations between study IDs and image IDs.
- Check whether each linked image path exists locally after data access is
  available.
- Track dataset availability flags for MIMIC-CXR-JPG, Chest ImaGenome, and
  RadGraph.
- Do not commit raw image paths containing restricted local data dumps if they
  expose patient-linked records.

## 5. Cross-Source Consistency Checklist

- Count studies present in all available sources.
- Compare Chest ImaGenome finding labels with RadGraph observations at the study
  level.
- Mark disagreement, uncertain, missing, and conflict states separately.
- Do not treat absent labels from one source as negative evidence.
- Exclude conflicting or ambiguous samples from the first pilot.

## 6. G1/G2/G3a/G3b Candidate Count Checklist

- Count valid G1 finding-existence items.
- Count valid G2 anatomy-finding pairs.
- Count possible G3a contrastive pairs using target anatomy plus plausible
  distractor anatomy.
- Count G3b candidates with positive findings and non-empty modifier evidence.
- Count candidates excluded because evidence is missing, uncertain, conflicting,
  or not phrase-supported.
- Keep G3b candidates marked as weak/silver unless manually validated.

## 7. Manual Sample Inspection Checklist

- Sample 30 phrases and check whether each phrase supports the structured label.
- Inspect examples from frequent findings and rare but clinically important
  findings.
- Check whether modifier cues bind to the intended finding.
- Record ambiguous cases and exclusion reasons.

## 8. Bbox Visualization Checklist

- Visualize 20 anatomical bboxes on images after MIMIC-CXR-JPG access is ready.
- Check coordinate alignment against the original image size.
- Confirm that bboxes represent anatomical regions, not lesion masks.
- Flag invalid, shifted, missing, or ambiguous boxes.

## 9. First-Pilot Go/No-Go Criteria

Proceed to first pilot only if:

- Linked image/report/annotation coverage is sufficient for a small pilot.
- G1 and G2 candidates have explicit evidence and clean exclusion handling.
- G3a distractors are anatomically plausible and not ambiguous.
- G3b candidates have phrase-level support or are clearly marked weak/silver.
- Conflicting and ambiguous samples are excluded from the first pilot.
- At least a small manual inspection batch supports the construction rules.
- Bbox coordinate alignment is acceptable for any bbox-dependent probe.
