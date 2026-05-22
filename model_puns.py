"""Parser for MODEL.txt — hand-picked model puns, NOT AI-processed.

Strict syntax (matches what MODEL.txt's header documents):

    <text>                          # text only
    <text> | <rating 1-10>          # text + rating
    <text> | <rating 1-10> | <comment>

`\\|` inside the text is an escaped pipe and stays as `|` in the parsed
output. Lines starting with `#` and blank lines are skipped.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

from settings import load_settings


@dataclass
class ModelPun:
    text: str
    rating: Optional[float] = None     # 1-10 if provided
    comment: Optional[str] = None


_PIPE_SPLIT = re.compile(r"(?<!\\)\|")


def _split_fields(line: str) -> list[str]:
    parts = _PIPE_SPLIT.split(line)
    return [p.replace("\\|", "|").strip() for p in parts]


def load_model_puns(path: Optional[str] = None) -> list[ModelPun]:
    if path is None:
        path = load_settings()["model_puns_path"]
    if not os.path.exists(path):
        return []
    out: list[ModelPun] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = _split_fields(line)
            if not fields or not fields[0]:
                continue
            text = fields[0]
            rating: Optional[float] = None
            comment: Optional[str] = None
            if len(fields) >= 2 and fields[1]:
                try:
                    r = float(fields[1])
                    if 1.0 <= r <= 10.0:
                        rating = r
                    else:
                        print(
                            f"  ⚠ {path}:{lineno} rating out of range (1-10): {fields[1]!r}",
                            file=sys.stderr,
                        )
                except ValueError:
                    print(
                        f"  ⚠ {path}:{lineno} could not parse rating {fields[1]!r}",
                        file=sys.stderr,
                    )
            if len(fields) >= 3 and fields[2]:
                comment = fields[2]
            out.append(ModelPun(text=text, rating=rating, comment=comment))
    return out


if __name__ == "__main__":
    for p in load_model_puns():
        print(p)
