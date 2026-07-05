"""Study 2 claim-verification constants and helpers."""

from __future__ import annotations

from typing import Any

from ghm.granularity.common import stable_item_id


QUESTION_TYPE = "claim_verification_abc"
PROMPT_TEMPLATE_ID = "claim_verification_abc_v1"
MISSING_FINDING_SAMPLE_SIZE = 2

ANSWER_SUPPORTED = "Supported"
ANSWER_CONTRADICTED = "Contradicted"
ANSWER_NOT_ENOUGH = "Not enough evidence"

CLAIM_POSITIVE = "positive"
CLAIM_NEGATIVE = "negative"

EVIDENCE_AFFIRMED = "affirmed"
EVIDENCE_NEGATED = "negated"
EVIDENCE_NOT_ENOUGH = "not_enough_evidence"

ANATOMICAL_FINDING_VOCAB = [
    "lung opacity",
    "airspace opacity",
    "consolidation",
    "infiltration",
    "atelectasis",
    "linear/patchy atelectasis",
    "lobar/segmental collapse",
    "pulmonary edema/hazy opacity",
    "vascular congestion",
    "vascular redistribution",
    "increased reticular markings/ild pattern",
    "pleural effusion",
    "costophrenic angle blunting",
    "pleural/parenchymal scarring",
    "bronchiectasis",
    "enlarged cardiac silhouette",
    "mediastinal displacement",
    "mediastinal widening",
    "enlarged hilum",
    "tortuous aorta",
    "vascular calcification",
    "pneumomediastinum",
    "pneumothorax",
    "hydropneumothorax",
    "lung lesion",
    "mass/nodule (not otherwise specified)",
    "multiple masses/nodules",
    "calcified nodule",
    "superior mediastinal mass/enlargement",
    "rib fracture",
    "clavicle fracture",
    "spinal fracture",
    "hyperaeration",
    "cyst/bullae",
    "elevated hemidiaphragm",
    "diaphragmatic eventration (benign)",
    "subdiaphragmatic air",
    "subcutaneous air",
    "hernia",
    "scoliosis",
    "spinal degenerative changes",
    "shoulder osteoarthritis",
    "bone lesion",
]


def answer_for_claim(*, evidence_state: str, claim_polarity: str) -> str:
    """Map evidence state and claim polarity to the Study 2 answer label."""

    if evidence_state == EVIDENCE_NOT_ENOUGH:
        return ANSWER_NOT_ENOUGH
    if evidence_state == EVIDENCE_AFFIRMED:
        return ANSWER_SUPPORTED if claim_polarity == CLAIM_POSITIVE else ANSWER_CONTRADICTED
    if evidence_state == EVIDENCE_NEGATED:
        return ANSWER_CONTRADICTED if claim_polarity == CLAIM_POSITIVE else ANSWER_SUPPORTED
    raise ValueError(f"Unknown evidence_state: {evidence_state}")


def claim_text(finding: str, *, claim_polarity: str, anatomy: str | None) -> str:
    """Render a Study 2 positive or negative medical claim."""

    if anatomy is None:
        target = "in this chest X-ray"
    else:
        target = f"in the {anatomy}"
    if claim_polarity == CLAIM_POSITIVE:
        return f"There is evidence of {finding} {target}."
    if claim_polarity == CLAIM_NEGATIVE:
        return f"There isn't evidence of {finding} {target}."
    raise ValueError(f"Unknown claim_polarity: {claim_polarity}")


def question_for_claim(claim: str) -> str:
    """Render the model-facing Study 2 claim verification question."""

    return (
        "Evaluate the following medical claim based only on the visible "
        f"radiographic evidence. Claim: {claim}"
    )


def stable_missing_findings(
    *,
    mentioned: set[str],
    sample_size: int,
    seed: int,
    scope_components: dict[str, Any],
) -> list[str]:
    """Choose unmentioned findings from the fixed vocabulary deterministically."""

    return stable_missing_findings_from_vocab(
        vocabulary=ANATOMICAL_FINDING_VOCAB,
        mentioned=mentioned,
        sample_size=sample_size,
        seed=seed,
        scope_components=scope_components,
    )


def stable_missing_findings_from_vocab(
    *,
    vocabulary: list[str] | set[str],
    mentioned: set[str],
    sample_size: int,
    seed: int,
    scope_components: dict[str, Any],
) -> list[str]:
    """Choose unmentioned findings from a supplied vocabulary deterministically."""

    missing = sorted(finding for finding in vocabulary if finding not in mentioned)
    ranked = sorted(
        missing,
        key=lambda finding: stable_item_id(
            "study2_missing",
            {"seed": seed, "finding": finding, **scope_components},
        ),
    )
    return ranked[:sample_size]
