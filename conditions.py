"""Loader for CONDITIONS.txt — the human-authored brief describing what
makes a good pun. Plain text; loaded verbatim into the generator and
judge system prompts.
"""

from __future__ import annotations

import os

from settings import load_settings


_FALLBACK = (
    "A good pun is a complete utterance that carries meaning in its "
    "discourse while activating two readings of an ambiguous pivot. "
    "Both readings should be plausible at once, anchored by distinct "
    "context cues, with the second reading landing as a small surprise."
)


def load_conditions(path: str | None = None) -> str:
    if path is None:
        path = load_settings()["conditions_path"]
    if not os.path.exists(path):
        return _FALLBACK
    with open(path) as f:
        text = f.read().strip()
    return text or _FALLBACK
