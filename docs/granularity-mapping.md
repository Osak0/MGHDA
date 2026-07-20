# Granularity Mapping

This is the source of truth for the current optimized Study 2 experiment. The
taxonomy names remain stable; only G1 and G2 are runnable in this repository.

Study 3 reuses the same `G1_finding_existence` and
`G2_anatomical_localization` taxonomy definitions without redefining them. Its
independent explicit-finding multi-select task, construction rules, schemas,
and metrics are defined in `docs/study3-multiselect-construction.md`.

## Study 2 common task

G1 and G2 use `claim_verification_abc` with template
`claim_verification_abc_v1` and exactly three labels:

```text
A. Supported
B. Contradicted
C. Not enough evidence
```

For each selected finding, construct both positive and negative claims. Explicit
`yes` and `no` assertions determine Supported versus Contradicted. A missing
assertion is not negative evidence and maps to Not enough evidence for both
claim polarities. `Unknown` and `Conflict` remain separate states; conflicts are
excluded and counted for the first run.

## G1_finding_existence

- Unit: finding existence across the whole image.
- Evidence: Chest ImaGenome `anatomicalfinding|yes/no|{finding}` assertions from
  all attribute dictionaries, including attributes without a usable bbox.
- Positive claim: `There is evidence of {finding} in this chest X-ray.`
- Negative claim: `There isn't evidence of {finding} in this chest X-ray.`
- Missing probes: sample two findings per image from the fixed Chest ImaGenome
  anatomical-finding vocabulary with seed 42, excluding findings mentioned in
  the image.

## G2_anatomical_localization

- Unit: finding existence in one named anatomy region.
- Evidence requires `anatomy_bound=True`, a real `bbox_name`, and a matching
  Chest ImaGenome object bbox. `bbox_name=False` is never localization evidence;
  an anatomical bbox is not a lesion segmentation.
- Positive claim: `There is evidence of {finding} in the {bbox_name}.`
- Negative claim: `There isn't evidence of {finding} in the {bbox_name}.`
- Missing probes use only findings previously observed for the same `bbox_name`
  in `bbox_finding_vocab_summary.csv`.
- Exclude findings mentioned in the current image+bbox and findings positive
  elsewhere in the same image.
- Create missing probes only when the bbox audit recommendation is `use_for_g2`.
  Bboxes marked `use_explicit_only`, `review`, or `drop_for_g2` may retain valid
  explicit claims but receive no missing probes.

## Scoring interpretation

- Ground truth Contradicted, model Supported: `H1_evidence_contradicted`.
- Ground truth Not enough evidence, model Supported:
  `H2_evidence_unsupported`.
- Other wrong ABC answers: `incorrect_non_hallucination`.
- Empty, unparsable, or multiple answers: `invalid_response`; these are reported
  separately and are not infrastructure failures.

Aggregate results are reported overall and by granularity, claim polarity,
evidence state, ground-truth label, and parsed answer.

## Reserved taxonomy

`G3a_contrastive_finding_anatomy_binding`,
`G3b_modifier_characterization`, `G3c_temporal_or_comparison_relation`, and
`G4_structured_synthesis_proxy` remain defined research directions but are not
implemented or run in this migration.
