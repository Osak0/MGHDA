# Study 2 Prompt Construction

This document is the experiment specification for the optimized Study 2 G1/G2
claim-verification run. It supersedes the former root-level
`prompt-construct3.md` draft.

## Answer space

Every item uses exactly one answer:

```text
A. Supported
B. Contradicted
C. Not enough evidence
```

The model must return only `A`, `B`, or `C`.

## Claim construction

For every explicit Chest ImaGenome anatomical finding, construct a positive and
a negative claim. G1 uses `in this chest X-ray`; G2 uses `in the {bbox_name}`.

- Source `yes`: the positive claim is Supported and the negative claim is
  Contradicted.
- Source `no`: the positive claim is Contradicted and the negative claim is
  Supported.
- An unmentioned sampled finding is missing evidence. Both claim polarities are
  labeled Not enough evidence; missing labels are never converted to `No`.
- Conflicting explicit assertions are excluded and counted.

## Optimized missing-evidence sampling

- Use `MISSING_FINDING_SAMPLE_SIZE=2` and `MISSING_FINDING_SEED=42` for the
  reference run.
- G1 samples from the fixed Chest ImaGenome anatomical-finding vocabulary after
  excluding findings mentioned in the image.
- G2 samples only from findings observed with the same `bbox_name` in the
  aggregate bbox/finding reference table.
- G2 excludes findings mentioned in the current image+bbox and findings that are
  positive anywhere else in the same image.
- G2 missing probes are created only for bbox names classified `use_for_g2` by
  the aggregate coordinate-quality audit. Other bbox names retain explicit
  claims but do not receive missing-evidence probes.
- A Chest ImaGenome attribute with `bbox_name=False` is never used as G2
  localization evidence.

## Reference experiment sampling

- Build the complete deterministic candidate pool first, then retain only
  candidates whose MIMIC-CXR-JPG image already exists locally.
- The reference run uses `STUDY2_MAX_ITEMS_TOTAL=960` and
  `STUDY2_SAMPLE_SEED=42`: 480 items each for G1 and G2.
- Sampling keeps the positive and negative claims for one evidence target as an
  inseparable pair. Each granularity selects 80 pairs for each of affirmed,
  negated, and not-enough-evidence states. This yields 160 Supported, 160
  Contradicted, and 160 Not enough evidence items per granularity.
- If an evidence stratum has fewer eligible local-image pairs than its quota,
  keep the smaller count and report the shortage; never substitute a missing
  image or convert missing evidence to a negative label.

## Separated prompt layers

Model-input JSONL contains only `item_id`, `image_path`, `prompt_template_id`,
and `prompt`. Ground-truth labels and structured evidence are stored separately
in eval-metadata JSONL and joined by `item_id`. Both files and all image paths
are restricted experiment artifacts and must never be committed to Git.
`image_path` is portable and relative to `MGHDA_DATA_ROOT`, for example
`files/p10/.../image.jpg`.
