"""Pun corpus.

A JSON-backed store of the puns that have made it past the judge. Each entry
records the original pun, the generator's score, the judge's verdict, and a
`verified` flag set when a human was involved in the verdict (judge source
`human` or `hybrid`). LLM-only verdicts are stored UNVERIFIED.

The corpus is the ground truth that the learnlog summarises.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional


DEFAULT_CORPUS_PATH = os.environ.get("PUN_CORPUS_PATH", "corpus.json")
DEFAULT_ARCHIVE_PATH = os.environ.get("PUN_ARCHIVE_PATH", "archive.txt")

# A pun must clear this judge overall score to enter the corpus by default.
DEFAULT_THRESHOLD = 0.65


@dataclass
class CorpusEntry:
    pun: dict                  # full Pun fields
    generator_score: dict      # PunScore fields
    judge: dict                # JudgeVerdict fields
    verified: bool             # True iff a human contributed to the verdict
    topic: str
    audience: str
    tone: str
    added_at: str              # ISO 8601 UTC


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def load_corpus(path: str = DEFAULT_CORPUS_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_corpus(corpus: list[dict], path: str = DEFAULT_CORPUS_PATH) -> None:
    with open(path, "w") as f:
        json.dump(corpus, f, indent=2)


# ---------------------------------------------------------------------------
# Adding entries
# ---------------------------------------------------------------------------


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def _is_verified(judge: dict) -> bool:
    return judge.get("source") in ("human", "hybrid")


def _archive_below_threshold(
    corpus: list[dict],
    threshold: float,
    archive_path: str,
    *,
    reason: str = "threshold-change",
) -> list[dict]:
    """Move corpus entries whose judge overall < threshold into the archive
    (JSON Lines, append-only). Returns the trimmed corpus."""
    keep, demote = [], []
    for e in corpus:
        if e["judge"]["overall"] < threshold:
            demote.append(e)
        else:
            keep.append(e)
    if demote:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(archive_path, "a") as f:
            for e in demote:
                f.write(json.dumps({
                    "archived_at": now,
                    "reason": reason,
                    "threshold": threshold,
                    "entry": e,
                }) + "\n")
    return keep


def add_to_corpus(
    judged: list[dict],
    topic: str,
    audience: str,
    tone: str = "neutral",
    *,
    threshold: float = DEFAULT_THRESHOLD,
    path: str = DEFAULT_CORPUS_PATH,
    archive_path: str = DEFAULT_ARCHIVE_PATH,
    max_entries: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Merge `judged` rows (output of `judge.judge_puns`) into the corpus.

    A row is admitted if `judge.overall >= threshold`. Duplicates (same pun
    text, case-insensitive) are merged: an UNVERIFIED entry is upgraded to
    VERIFIED if the new verdict came from a human, and the higher overall
    score wins.

    Existing corpus entries that now fall below `threshold` (e.g. because
    the threshold was raised in SETTINGS.txt) are demoted into the archive.

    Returns (full_corpus, newly_added_or_upgraded_entries).
    """
    corpus = load_corpus(path)
    # Reconcile against the current threshold BEFORE merging new entries.
    corpus = _archive_below_threshold(corpus, threshold, archive_path)
    by_key: dict[str, dict] = {_key(e["pun"]["text"]): e for e in corpus}
    changes: list[dict] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for row in judged:
        if row["judge"]["overall"] < threshold:
            continue
        verified = _is_verified(row["judge"])
        entry = asdict(CorpusEntry(
            pun=row["pun"],
            generator_score=row["generator_score"],
            judge=row["judge"],
            verified=verified,
            topic=topic,
            audience=audience,
            tone=tone,
            added_at=now,
        ))
        k = _key(entry["pun"]["text"])
        existing = by_key.get(k)
        if existing is None:
            by_key[k] = entry
            changes.append(entry)
            continue
        # Merge: upgrade verification, keep best overall, refresh judge
        upgraded = (not existing["verified"]) and verified
        better = entry["judge"]["overall"] > existing["judge"]["overall"]
        if upgraded or better:
            merged = {**existing}
            if upgraded:
                merged["verified"] = True
            if better:
                merged["judge"] = entry["judge"]
                merged["generator_score"] = entry["generator_score"]
            merged["added_at"] = now
            by_key[k] = merged
            changes.append(merged)

    merged_corpus = list(by_key.values())
    # Order: verified first, then by judge overall descending.
    merged_corpus.sort(
        key=lambda e: (e["verified"], e["judge"]["overall"]),
        reverse=True,
    )
    if max_entries is not None:
        merged_corpus = merged_corpus[:max_entries]

    save_corpus(merged_corpus, path)
    return merged_corpus, changes


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def stats(corpus: list[dict]) -> dict:
    if not corpus:
        return {"size": 0, "verified": 0, "unverified": 0}
    verified = sum(1 for e in corpus if e["verified"])
    return {
        "size": len(corpus),
        "verified": verified,
        "unverified": len(corpus) - verified,
        "mean_judge_overall": (
            sum(e["judge"]["overall"] for e in corpus) / len(corpus)
        ),
        "topics": sorted({e["topic"] for e in corpus}),
    }


def sample_exemplars(corpus: list[dict], k: int) -> list[dict]:
    """Return up to `k` exemplar puns. VERIFIED entries are preferred — if
    there are at least `k` of them, that's where the sample comes from.
    Otherwise verified entries are all included and the rest is filled with
    the highest-scoring unverified ones at random."""
    if k <= 0 or not corpus:
        return []
    verified = [e for e in corpus if e["verified"]]
    unverified = [e for e in corpus if not e["verified"]]
    if len(verified) >= k:
        chosen = random.sample(verified, k)
    else:
        unverified.sort(key=lambda e: e["judge"]["overall"], reverse=True)
        needed = k - len(verified)
        chosen = verified + random.sample(
            unverified, min(needed, len(unverified))
        )
    return [
        {
            "text": e["pun"]["text"],
            "pivot": e["pun"]["pivot"],
            "original": e["pun"]["original"],
            "topic": e.get("topic"),
            "audience": e.get("audience"),
            "tone": e.get("tone"),
            "verified": e["verified"],
            "judge_overall": round(e["judge"]["overall"], 2),
        }
        for e in chosen
    ]


def reconcile(
    threshold: float,
    path: str = DEFAULT_CORPUS_PATH,
    archive_path: str = DEFAULT_ARCHIVE_PATH,
) -> int:
    """Apply a (possibly new) threshold to the existing corpus: demote any
    entry below it to the archive. Returns the number of entries archived."""
    corpus = load_corpus(path)
    before = len(corpus)
    trimmed = _archive_below_threshold(corpus, threshold, archive_path)
    save_corpus(trimmed, path)
    return before - len(trimmed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or manage the pun corpus.")
    parser.add_argument("--path", default=DEFAULT_CORPUS_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="show summary stats")
    show = sub.add_parser("show", help="list entries")
    show.add_argument("--verified-only", action="store_true")
    show.add_argument("--limit", type=int, default=20)

    add = sub.add_parser("add", help="add judged JSON (from judge.py) into the corpus")
    add.add_argument("--input", "-i", default="-", help="path to judge JSON (or '-' for stdin)")
    add.add_argument("--topic", required=True)
    add.add_argument("--audience", required=True)
    add.add_argument("--tone", default="neutral")
    add.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    add.add_argument("--max-entries", type=int, default=None)

    rec = sub.add_parser(
        "reconcile",
        help="apply current threshold; demote sub-threshold entries to archive.txt",
    )
    rec.add_argument("--threshold", type=float, default=None,
                     help="threshold to apply (defaults to SETTINGS.txt)")

    args = parser.parse_args()

    if args.cmd == "stats":
        print(json.dumps(stats(load_corpus(args.path)), indent=2))
        return

    if args.cmd == "show":
        corpus = load_corpus(args.path)
        if args.verified_only:
            corpus = [e for e in corpus if e["verified"]]
        for e in corpus[: args.limit]:
            tag = "VERIFIED" if e["verified"] else "UNVERIFIED"
            print(f"[{tag}] {e['judge']['overall']:.2f}  {e['pun']['text']}")
            print(f"          topic={e['topic']!r}  audience={e['audience']!r}")
        return

    if args.cmd == "add":
        import sys
        try:
            if args.input == "-":
                judged = json.load(sys.stdin)
            else:
                with open(args.input) as f:
                    judged = json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(
                f"corpus add: input is not valid JSON ({e}). "
                "Make sure the upstream command was run with --format json, "
                "or use pun.py for the one-shot pipeline."
            )
        _, changes = add_to_corpus(
            judged, args.topic, args.audience, args.tone,
            threshold=args.threshold,
            path=args.path,
            max_entries=args.max_entries,
        )
        print(json.dumps({"changes": len(changes)}, indent=2))
        return

    if args.cmd == "reconcile":
        threshold = args.threshold
        if threshold is None:
            from settings import load_settings
            threshold = float(load_settings()["threshold"])
        n = reconcile(threshold, path=args.path)
        print(json.dumps({"archived": n, "threshold": threshold}, indent=2))


if __name__ == "__main__":
    _main()
