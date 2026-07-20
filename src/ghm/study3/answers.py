"""Strict parser for Study 3 multiple-select answers."""

from __future__ import annotations

import re
from typing import Iterable


OPTION_PATTERN = re.compile(r"^[A-Z]+$")
NONE_PATTERN = re.compile(r"^NONE[.!]?$", re.IGNORECASE)


def parse_multiselect_answer(
    raw_response: object,
    valid_option_ids: Iterable[str],
) -> tuple[list[str] | None, str]:
    """Parse comma/space/``and`` separated option IDs or ``NONE``."""

    ordered_valid_ids = [str(value).upper() for value in valid_option_ids]
    valid_ids = set(ordered_valid_ids)
    if not isinstance(raw_response, str) or not raw_response.strip():
        return None, "invalid_empty"
    text = raw_response.strip().upper()
    if NONE_PATTERN.fullmatch(text):
        return [], "valid_none"
    if re.search(r"\bNONE\b", text):
        return None, "invalid_none_with_options"

    normalized = text
    if normalized[:1] in "[(" and normalized[-1:] in "])":
        normalized = normalized[1:-1].strip()
    normalized = re.sub(r"\bAND\b", ",", normalized)
    normalized = normalized.replace(";", ",")
    pieces = [
        piece
        for chunk in normalized.split(",")
        for piece in chunk.strip().split()
        if piece
    ]
    if not pieces or any(not OPTION_PATTERN.fullmatch(piece) for piece in pieces):
        return None, "invalid_format"
    if len(pieces) != len(set(pieces)):
        return None, "invalid_duplicate_option"
    unknown = set(pieces) - valid_ids
    if unknown:
        return None, "invalid_unknown_option"
    order = {option_id: index for index, option_id in enumerate(ordered_valid_ids)}
    return sorted(pieces, key=lambda value: order[value]), "valid_options"
