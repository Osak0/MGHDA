"""Constants and isolation guards for Study 3."""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_ID = "study3_multiselect_v2"
QUESTION_TYPE = "anatomicalfinding_multiselect_v2"
PRESENT = "present"
ABSENT = "absent"
QUERY_RELATIONS = (PRESENT, ABSENT)
STATE = "state"
EVIDENCE = "evidence"
PROMPT_FRAMINGS = (STATE, EVIDENCE)
NATURAL = "natural"
CONTROLLED = "controlled"
VARIANTS = (NATURAL, CONTROLLED)
DEFAULT_NATURAL_ANCHORS = 250
DEFAULT_CONTROLLED_ANCHORS = 50
DEFAULT_CONTROLLED_K = (2, 3, 4, 5)
DEFAULT_SEED = 42


def require_study3_experiment(value: object) -> None:
    """Reject missing or cross-experiment records."""

    if value != EXPERIMENT_ID:
        raise ValueError(
            f"Study 3 requires experiment_id={EXPERIMENT_ID!r}; received {value!r}"
        )


def require_study3_item_id(value: object) -> None:
    """Reject item IDs that can collide with another experiment."""

    if not isinstance(value, str) or not value.startswith("study3_"):
        raise ValueError(f"Study 3 item_id must start with 'study3_'; received {value!r}")


def require_study3_output_path(path: Path) -> None:
    """Prevent Study 3 commands from writing to Study 2-named artifacts."""

    parts = [part.lower() for part in path.parts]
    if any(part.startswith("study2") for part in parts):
        raise ValueError(f"Study 3 refuses Study 2 output path: {path}")
    if "study3" not in parts and "study3" not in path.name.lower():
        raise ValueError(f"Study 3 output path must use a Study 3 namespace: {path}")
