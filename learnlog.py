"""Pun learn-log.

A rolling, auditable list of AI-written "lessons" derived from the corpus.
Every time the corpus changes, a fresh summary is generated from the
current corpus (preferring VERIFIED entries) and the previous summary,
and appended as a new entry. Only the latest summary is ever fed back
into the generator — the older entries exist for traceability, not for
prompt input.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from corpus import DEFAULT_CORPUS_PATH, load_corpus, stats
from llm import LLMClient
from model_puns import load_model_puns
from pun_generator import _strip_fences
from safety import safe_complete


DEFAULT_LEARNLOG_PATH = os.environ.get("PUN_LEARNLOG_PATH", "learnlog.json")

# How many corpus entries to feed into the summariser. The corpus itself
# may grow unbounded; the summariser only needs a representative sample,
# verified-first.
SUMMARY_SAMPLE_SIZE = 40


@dataclass
class LearnLogEntry:
    summary: str
    created_at: str          # ISO 8601 UTC
    corpus_size: int
    verified_count: int


SUMMARY_SYSTEM = """You are the meta-learner for a pun-generation system.

You will see three inputs:
  1. MODEL puns — hand-picked exemplars curated by the project owner. These
     are the GOLD STANDARD: the generator should aspire to write puns that
     match their craft. Some carry an optional rating and comment.
  2. CORPUS sample — recently-judged puns, each marked VERIFIED (human in
     the loop) or UNVERIFIED (LLM-only judge). Use these to spot what the
     generator currently produces, and how far it is from the model puns.
  3. The previous lessons document, if any.

Your job: produce a fresh "lessons" document — 200-400 words, flowing prose,
imperative voice ("Pick pivots that…", "Avoid building context only from…").
It will be APPENDED to the generator's system prompt next time it runs, so
token budget matters.

Anchor the lessons on the MODEL puns. The CORPUS sample is evidence about
what the generator does now; compare to the MODEL to identify the gap and
write rules that close it. Weight VERIFIED corpus entries above UNVERIFIED.

Cover, where the evidence supports it:
  - Concrete patterns in the strongest puns: pivot shape, soundalike
    distance, how the two readings are anchored, sentence cadence.
  - Failure modes worth avoiding: shared-context puns, lopsided readings,
    pivots that don't actually sound alike, off-topic filler.
  - Audience / topic / tone-specific cues if any pattern is visible.

Do NOT list individual puns. Generalise into rules. Carry forward useful
guidance from the previous document, dropping anything the new corpus
contradicts.

Return ONLY the prose. No headers, no bullets, no preamble, no JSON.
"""


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def load_learnlog(path: str = DEFAULT_LEARNLOG_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_learnlog(entries: list[dict], path: str = DEFAULT_LEARNLOG_PATH) -> None:
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def latest_summary(path: str = DEFAULT_LEARNLOG_PATH) -> Optional[str]:
    """The single piece that gets fed to the generator. None if no log yet."""
    entries = load_learnlog(path)
    return entries[-1]["summary"] if entries else None


# ---------------------------------------------------------------------------
# Sampling the corpus for the summariser
# ---------------------------------------------------------------------------


def _sample_corpus_for_summary(corpus: list[dict], k: int = SUMMARY_SAMPLE_SIZE) -> list[dict]:
    verified = [e for e in corpus if e["verified"]]
    unverified = [e for e in corpus if not e["verified"]]
    verified.sort(key=lambda e: e["judge"]["overall"], reverse=True)
    unverified.sort(key=lambda e: e["judge"]["overall"], reverse=True)
    # Verified first, then top unverified to fill the budget.
    chosen = verified[:k] + unverified[: max(0, k - len(verified))]
    # Trim each entry down to the fields the summariser actually needs.
    return [
        {
            "text": e["pun"]["text"],
            "pivot": e["pun"]["pivot"],
            "original": e["pun"]["original"],
            "topic": e["topic"],
            "audience": e["audience"],
            "verified": e["verified"],
            "judge_overall": round(e["judge"]["overall"], 2),
            "judge_breakdown": {
                k: round(v, 2) for k, v in e["judge"].items()
                if k in ("phonetic", "ambiguity", "distinctiveness", "surprise", "relevance")
                and v is not None
            },
        }
        for e in chosen
    ]


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(
    client: "LLMClient",
    corpus: list[dict],
    previous_summary: Optional[str] = None,
) -> str:
    """Generate a fresh lessons document. Pulls MODEL.txt puns as the gold
    standard and the corpus as the current state of practice."""
    model_pun_objs = load_model_puns()
    model_block = [
        {
            "text": m.text,
            "rating": m.rating,
            "comment": m.comment,
        }
        for m in model_pun_objs
    ]
    if not corpus and not model_block:
        return ""
    payload = {
        "model_puns_gold_standard": model_block,
        "corpus_sample": _sample_corpus_for_summary(corpus),
        "previous_lessons": previous_summary or "(none — this is the first summary)",
    }
    text = safe_complete(
        client,
        model=client.generator_model,
        max_tokens=800,
        system=[{"type": "text", "text": SUMMARY_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        user=json.dumps(payload, indent=2),
        json_mode=False,  # summary is prose, not JSON
    )
    return _strip_fences(text)


def update_after_corpus_change(
    *,
    client: Optional["LLMClient"] = None,
    corpus_path: str = DEFAULT_CORPUS_PATH,
    learnlog_path: str = DEFAULT_LEARNLOG_PATH,
) -> Optional[dict]:
    """Generate and append a new summary based on the current corpus.

    Returns the new entry, or None if the corpus is empty.
    """
    corpus = load_corpus(corpus_path)
    model_pun_objs = load_model_puns()
    if not corpus and not model_pun_objs:
        return None
    if client is None:
        client = LLMClient()

    prev = latest_summary(learnlog_path)
    summary = generate_summary(client, corpus, prev)
    if not summary:
        return None
    s = stats(corpus)
    entry = asdict(LearnLogEntry(
        summary=summary,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        corpus_size=s["size"],
        verified_count=s["verified"],
    ))
    log = load_learnlog(learnlog_path)
    log.append(entry)
    save_learnlog(log, learnlog_path)
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or refresh the learnlog.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--learnlog", default=DEFAULT_LEARNLOG_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("refresh", help="generate a new summary from the current corpus")
    sub.add_parser("latest", help="print the latest summary (what gets fed to the generator)")
    history = sub.add_parser("history", help="list all summaries")
    history.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "refresh":
        entry = update_after_corpus_change(
            corpus_path=args.corpus, learnlog_path=args.learnlog
        )
        if entry is None:
            print("corpus is empty; no summary generated")
            return
        print(entry["summary"])
        return

    if args.cmd == "latest":
        s = latest_summary(args.learnlog)
        print(s if s is not None else "(no summary yet — refresh first)")
        return

    if args.cmd == "history":
        log = load_learnlog(args.learnlog)
        for e in log[-args.limit:]:
            print(f"── {e['created_at']}  (corpus={e['corpus_size']}, verified={e['verified_count']})")
            print(e["summary"])
            print()


if __name__ == "__main__":
    _main()
