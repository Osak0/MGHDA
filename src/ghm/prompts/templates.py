"""Define prompt templates for model-ready item rendering."""

from __future__ import annotations


YES_NO_UNCERTAIN_TEMPLATE_ID = "yes_no_uncertain_v1"
YES_NO_UNCERTAIN_SYSTEM = (
    "You are given a chest X-ray. "
    "Answer only one of: Yes, No, or Uncertain."
)
SUPPORTED_UNSUPPORTED_TEMPLATE_ID = "supported_unsupported_uncertain_v1"
SUPPORTED_UNSUPPORTED_SYSTEM = (
    "You are given a chest X-ray. "
    "Answer only one of: Supported, Unsupported, or Uncertain."
)


def render_yes_no_uncertain_prompt(question: str) -> str:
    """Render a closed-form medical VQA prompt."""

    return f"{YES_NO_UNCERTAIN_SYSTEM} Question: {question}"


def render_supported_unsupported_prompt(question: str) -> str:
    """Render a claim-support medical VQA prompt."""

    return f"{SUPPORTED_UNSUPPORTED_SYSTEM} Question: {question}"


def template_for_probe(hallucination_probe: str | None) -> tuple[str, str]:
    """Return template id and renderer system for a hallucination probe."""

    if hallucination_probe == "H2":
        return SUPPORTED_UNSUPPORTED_TEMPLATE_ID, SUPPORTED_UNSUPPORTED_SYSTEM
    return YES_NO_UNCERTAIN_TEMPLATE_ID, YES_NO_UNCERTAIN_SYSTEM
