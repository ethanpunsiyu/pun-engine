"""Loader for SETTINGS.txt.

Strict key=value syntax, one per line, no quoting. Lines starting with `#`
or that are blank are ignored. Whitespace around `=` is fine. Unknown keys
are kept but emit a stderr warning.

The defaults below are the source of truth for which keys exist and what
type each one coerces to. Editing this file is how you add a new option.
"""

from __future__ import annotations

import os
import sys
from typing import Any


DEFAULT_SETTINGS_PATH = os.environ.get("PUN_SETTINGS_PATH", "SETTINGS.txt")


# (default value, coercer)
_SCHEMA: dict[str, tuple[Any, Any]] = {
    # LLM provider — "anthropic" or "openai". Generator/judge models default
    # to provider-appropriate names if left blank (resolved by llm.py).
    "provider":             ("anthropic", str),
    "generator_model":      ("",   str),
    "judge_model":          ("",   str),

    "exemplar_count":       (3,    int),
    "threshold":            (0.65, float),
    "over_factor":          (2,    int),
    "summary_sample_size":  (40,   int),
    "max_api_retries":      (6,    int),
    "corpus_path":          ("corpus.json", str),
    "learnlog_path":        ("learnlog.json", str),
    "archive_path":         ("archive.txt", str),
    "model_puns_path":      ("MODEL.txt", str),
    "conditions_path":      ("CONDITIONS.txt", str),
}


def defaults() -> dict[str, Any]:
    return {k: v for k, (v, _) in _SCHEMA.items()}


def load_settings(path: str = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    out = defaults()
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(
                    f"  ⚠ {path}:{lineno} ignored (no '='): {raw.rstrip()!r}",
                    file=sys.stderr,
                )
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key not in _SCHEMA:
                print(f"  ⚠ {path}:{lineno} unknown key {key!r}; keeping as string",
                      file=sys.stderr)
                out[key] = value
                continue
            _, coercer = _SCHEMA[key]
            try:
                out[key] = coercer(value)
            except (TypeError, ValueError):
                print(
                    f"  ⚠ {path}:{lineno} could not coerce {value!r} for {key} "
                    f"(expected {coercer.__name__}); using default {out[key]!r}",
                    file=sys.stderr,
                )
    return out


def write_default_settings_file(path: str = DEFAULT_SETTINGS_PATH) -> None:
    """Write a SETTINGS.txt populated with current defaults. Used once on
    first run if the file doesn't exist."""
    lines = [
        "# SETTINGS.txt",
        "# One setting per line, `key = value`. # marks comments.",
        "# Edit freely — this file is not AI-processed.",
        "",
    ]
    for k, (v, _) in _SCHEMA.items():
        lines.append(f"{k} = {v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import json
    print(json.dumps(load_settings(), indent=2))
