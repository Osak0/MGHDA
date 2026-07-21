"""Define prompt templates for model-ready item rendering."""

from __future__ import annotations

from ghm.granularity.study2 import PROMPT_TEMPLATE_ID, QUESTION_TYPE


CLAIM_VERIFICATION_TEMPLATE_ID = PROMPT_TEMPLATE_ID
CLAIM_VERIFICATION_V1_TEMPLATE_ID = PROMPT_TEMPLATE_ID
CLAIM_VERIFICATION_V2_TEMPLATE_ID = "claim_verification_abc_definitions_v2"
CLAIM_VERIFICATION_V1_SYSTEM = (
    "You are given a chest X-ray. "
    "Answer only one of: A. Supported, B. Contradicted, C. Not enough evidence. "
    "Return only A, B, or C."
)
CLAIM_VERIFICATION_V2_SYSTEM = """You are given a chest X-ray.
Answer only one of:

A. Supported:
The radiographic evidence affirms the exact claim.

B. Contradicted:
The radiographic evidence supports the logical opposite of the claim.

C. Not enough evidence:
The radiographic evidence establishes neither the claim nor its logical opposite.

Return only A, B, or C."""
CLAIM_VERIFICATION_SYSTEM = CLAIM_VERIFICATION_V1_SYSTEM

# Backward-compatible names for older imports.
YES_NO_UNCERTAIN_TEMPLATE_ID = CLAIM_VERIFICATION_TEMPLATE_ID
SUPPORTED_UNSUPPORTED_TEMPLATE_ID = CLAIM_VERIFICATION_TEMPLATE_ID


def render_claim_verification_prompt(
    question: str,
    *,
    template_version: str = "v1",
) -> str:
    """Render a Study 2 ABC claim-verification prompt."""

    _, system = claim_verification_template(template_version)
    return f"{system}\n\nQuestion: {question}"


def claim_verification_template(template_version: str) -> tuple[str, str]:
    """Return the immutable Study 2 v1 or definition-only v2 template."""

    if template_version == "v1":
        return CLAIM_VERIFICATION_V1_TEMPLATE_ID, CLAIM_VERIFICATION_V1_SYSTEM
    if template_version == "v2":
        return CLAIM_VERIFICATION_V2_TEMPLATE_ID, CLAIM_VERIFICATION_V2_SYSTEM
    raise ValueError("template_version must be 'v1' or 'v2'")


def render_yes_no_uncertain_prompt(question: str) -> str:
    """Backward-compatible wrapper for legacy call sites."""

    return render_claim_verification_prompt(question)


def render_supported_unsupported_prompt(question: str) -> str:
    """Backward-compatible wrapper for legacy call sites."""

    return render_claim_verification_prompt(question)


def template_for_question_type(question_type: str | None) -> tuple[str, str]:
    """Return template id and system message for a question type."""

    if question_type == QUESTION_TYPE:
        return CLAIM_VERIFICATION_TEMPLATE_ID, CLAIM_VERIFICATION_SYSTEM
    return CLAIM_VERIFICATION_TEMPLATE_ID, CLAIM_VERIFICATION_SYSTEM


def template_for_probe(hallucination_probe: str | None) -> tuple[str, str]:
    """Backward-compatible template lookup."""

    return CLAIM_VERIFICATION_TEMPLATE_ID, CLAIM_VERIFICATION_SYSTEM
