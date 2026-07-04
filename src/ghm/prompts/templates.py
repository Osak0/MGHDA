"""Define prompt templates for model-ready item rendering."""

from __future__ import annotations

from ghm.granularity.study2 import PROMPT_TEMPLATE_ID, QUESTION_TYPE


CLAIM_VERIFICATION_TEMPLATE_ID = PROMPT_TEMPLATE_ID
CLAIM_VERIFICATION_SYSTEM = (
    "You are given a chest X-ray. "
    "Answer only one of: A. Supported, B. Contradicted, C. Not enough evidence. "
    "Return only A, B, or C."
)

# Backward-compatible names for older imports.
YES_NO_UNCERTAIN_TEMPLATE_ID = CLAIM_VERIFICATION_TEMPLATE_ID
SUPPORTED_UNSUPPORTED_TEMPLATE_ID = CLAIM_VERIFICATION_TEMPLATE_ID


def render_claim_verification_prompt(question: str) -> str:
    """Render a Study 2 ABC claim-verification prompt."""

    return f"{CLAIM_VERIFICATION_SYSTEM} Question: {question}"


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
