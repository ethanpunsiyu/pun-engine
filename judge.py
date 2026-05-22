"""Pun judge.

Scores puns produced by `pun_generator.generate_puns` independently of the
generator, against the same rubric, in one of three modes:

    llm     — a different LLM model than the generator rates every dimension
              and an overall holistic score.
    human   — a person rates each pun: ONE overall 1-10 score plus an
              optional free-text comment. Dimensional scores are left blank.
    hybrid  — the LLM rates the four semantic dimensions; the human supplies
              only the overall rating and an optional comment.

The judge's verdict sits alongside the generator's own score so disagreement
between them can be surfaced (per-pun delta + batch-level mean delta).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Optional

from conditions import load_conditions
from corpus import DEFAULT_CORPUS_PATH, DEFAULT_THRESHOLD, add_to_corpus
from learnlog import DEFAULT_LEARNLOG_PATH, update_after_corpus_change
from llm import LLMClient
from pun_generator import (
    _strip_fences,
    generate_puns,
    phonetic_similarity,
)
from safety import safe_complete


# ---------------------------------------------------------------------------
# Verdict type
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    overall: float                          # 0..1
    source: str                             # "llm" | "human" | "hybrid"
    phonetic: Optional[float] = None
    ambiguity: Optional[float] = None
    distinctiveness: Optional[float] = None
    surprise: Optional[float] = None
    relevance: Optional[float] = None
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


JUDGE_SYSTEM = """You are an independent pun judge. You did NOT write these puns — rate them honestly.

For each pun in the input array, produce ONE entry. Return ALL entries
wrapped in a single JSON object with shape:

{
  "rows": [
    {
      "id": <int>,
      "ambiguity": <float 0..1>,
      "distinctiveness": <float 0..1>,
      "surprise": <float 0..1>,
      "relevance": <float 0..1>,
      "overall": <float 0..1>,
      "comment": "<one short sentence — name the strongest weakness>"
    },
    ...
  ]
}

Rubric:
  ambiguity        — both meanings of the pivot fit the sentence simultaneously.
  distinctiveness  — surrounding context anchors each meaning separately and in a
                     balanced way (not lopsided, not shared evidence).
  surprise         — the pivot is unexpected at its local slot yet sensible in
                     the global topic.
  relevance        — the utterance actually says something on-topic for the
                     stated audience; not filler.
  overall          — your holistic judgement — NOT a mechanical average. Does
                     the pun land? Be strict: average puns sit at 0.4–0.6.

Return ONLY the JSON object — no prose, no code fences.
"""


def judge_with_llm(
    client: "LLMClient",
    puns: list[dict],
    topic: str,
    audience: str,
) -> list[JudgeVerdict]:
    """Score every pun in one round-trip against the rubric."""
    payload = {
        "topic": topic,
        "audience": audience,
        "puns": [{"id": i, **p} for i, p in enumerate(puns)],
    }
    system_blocks = [
        {"type": "text", "text": JUDGE_SYSTEM, "cache_control": {"type": "ephemeral"}},
        {
            "type": "text",
            "text": "Conditions (what counts as a good pun):\n\n" + load_conditions(),
            "cache_control": {"type": "ephemeral"},
        },
    ]
    text = safe_complete(
        client,
        model=client.judge_model,
        max_tokens=2048,
        system=system_blocks,
        user=json.dumps(payload, indent=2),
    )
    parsed = json.loads(_strip_fences(text))
    rows = parsed["rows"] if isinstance(parsed, dict) and "rows" in parsed else parsed
    by_id = {row["id"]: row for row in rows if isinstance(row, dict) and "id" in row}

    verdicts: list[JudgeVerdict] = []
    for i, p in enumerate(puns):
        row = by_id.get(i)
        ph = phonetic_similarity(p["pivot"], p["original"])
        if row is None:
            verdicts.append(JudgeVerdict(
                overall=0.5, source="llm", phonetic=ph,
                comment="judge skipped this entry",
            ))
            continue
        verdicts.append(JudgeVerdict(
            overall=float(row["overall"]),
            source="llm",
            phonetic=ph,
            ambiguity=float(row["ambiguity"]),
            distinctiveness=float(row["distinctiveness"]),
            surprise=float(row["surprise"]),
            relevance=float(row["relevance"]),
            comment=row.get("comment"),
        ))
    return verdicts


# ---------------------------------------------------------------------------
# Human judge
# ---------------------------------------------------------------------------


_tty_in = None
_tty_out = None


def _interactive_streams():
    """Return (read, write) streams that talk to the user even when stdin
    is being used to pipe JSON into the script."""
    global _tty_in, _tty_out
    if _tty_in is not None:
        return _tty_in, _tty_out
    if sys.stdin.isatty():
        _tty_in, _tty_out = sys.stdin, sys.stderr
    else:
        try:
            _tty_in = open("/dev/tty", "r")
            _tty_out = open("/dev/tty", "w")
        except OSError as e:
            raise RuntimeError(
                "human/hybrid judging needs a terminal, but stdin is piped "
                "and /dev/tty is unavailable. Run inside a real terminal, or "
                "pass --mode llm."
            ) from e
    return _tty_in, _tty_out


def _ask(prompt: str) -> str:
    rd, wr = _interactive_streams()
    wr.write(prompt)
    wr.flush()
    line = rd.readline()
    if not line:
        raise EOFError("no interactive input available")
    return line.rstrip("\n")


def _prompt_overall_rating(prompt: str) -> float:
    """Read a 1-10 rating from the user and normalise to 0..1."""
    _, wr = _interactive_streams()
    while True:
        raw = _ask(prompt).strip()
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            wr.write("  please enter a number between 1 and 10\n")
            continue
        if not 1.0 <= v <= 10.0:
            wr.write("  out of range, expected 1-10\n")
            continue
        return v / 10.0


def _show_pun(idx: int, total: int, pun: dict) -> None:
    _, wr = _interactive_streams()
    wr.write(f"\n[{idx}/{total}] {pun['text']}\n")
    wr.write(f"   pivot: {pun['pivot']!r}  ~  {pun['original']!r}\n")
    wr.write(f"   readings: local={pun['meaning_local']} | global={pun['meaning_global']}\n")
    wr.flush()


def judge_with_human(puns: list[dict]) -> list[JudgeVerdict]:
    verdicts: list[JudgeVerdict] = []
    total = len(puns)
    for i, p in enumerate(puns, 1):
        _show_pun(i, total, p)
        overall = _prompt_overall_rating("   overall rating (1-10): ")
        comment = _ask("   comment (optional, enter to skip): ").strip() or None
        verdicts.append(JudgeVerdict(
            overall=overall,
            source="human",
            phonetic=phonetic_similarity(p["pivot"], p["original"]),
            comment=comment,
        ))
    return verdicts


# ---------------------------------------------------------------------------
# Hybrid: LLM does the dimensions, human supplies overall + comment
# ---------------------------------------------------------------------------


def judge_hybrid(
    client: "LLMClient",
    puns: list[dict],
    topic: str,
    audience: str,
) -> list[JudgeVerdict]:
    llm_verdicts = judge_with_llm(client, puns, topic, audience)
    total = len(puns)
    out: list[JudgeVerdict] = []
    _, wr = _interactive_streams()
    for i, (p, v) in enumerate(zip(puns, llm_verdicts), 1):
        _show_pun(i, total, p)
        wr.write(
            f"   llm view: amb={v.ambiguity:.2f}  dist={v.distinctiveness:.2f}  "
            f"surp={v.surprise:.2f}  rel={v.relevance:.2f}  phon={v.phonetic:.2f}\n"
        )
        if v.comment:
            wr.write(f"   llm note: {v.comment}\n")
        wr.flush()
        overall = _prompt_overall_rating("   your overall rating (1-10): ")
        comment = _ask("   your comment (optional, enter to skip): ").strip() or None
        out.append(JudgeVerdict(
            overall=overall,
            source="hybrid",
            phonetic=v.phonetic,
            ambiguity=v.ambiguity,
            distinctiveness=v.distinctiveness,
            surprise=v.surprise,
            relevance=v.relevance,
            comment=comment,
        ))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_puns(
    generator_output: list[dict],
    mode: str,
    topic: str,
    audience: str,
    tone: str = "neutral",
    *,
    client: Optional["LLMClient"] = None,
    persist: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    corpus_path: str = DEFAULT_CORPUS_PATH,
    learnlog_path: str = DEFAULT_LEARNLOG_PATH,
    refresh_learnlog: bool = True,
) -> list[dict]:
    """Judge `generator_output` (as returned by `pun_generator.generate_puns`).

    Side effect (on by default): qualifying puns are written to the corpus
    and the learnlog is refreshed in the same call. Set `persist=False` to
    judge without touching disk, or `refresh_learnlog=False` to write to the
    corpus but skip the summary regeneration.

    Returns one row per pun:
        {"pun": {...}, "generator_score": {...}, "judge": {...}, "delta": float}
    where `delta = judge.overall - generator_score.overall` (positive = judge
    was more generous than the generator).
    """
    if not generator_output:
        return []
    puns = [row["pun"] for row in generator_output]

    if mode == "llm":
        if client is None:
            client = LLMClient()
        verdicts = judge_with_llm(client, puns, topic, audience)
    elif mode == "human":
        verdicts = judge_with_human(puns)
    elif mode == "hybrid":
        if client is None:
            client = LLMClient()
        verdicts = judge_hybrid(client, puns, topic, audience)
    else:
        raise ValueError(f"unknown mode: {mode!r}; expected llm/human/hybrid")

    judged = [
        {
            "pun": row["pun"],
            "generator_score": row["score"],
            "judge": asdict(v),
            "delta": v.overall - row["score"]["overall"],
        }
        for row, v in zip(generator_output, verdicts)
    ]

    if persist:
        _, changes = add_to_corpus(
            judged, topic, audience, tone,
            threshold=threshold, path=corpus_path,
        )
        print(
            f"▸ corpus: {len(changes)} entry/entries added or upgraded "
            f"(threshold={threshold})",
            file=sys.stderr,
        )
        if changes and refresh_learnlog:
            if client is None:
                client = LLMClient()
            entry = update_after_corpus_change(
                client=client,
                corpus_path=corpus_path,
                learnlog_path=learnlog_path,
            )
            if entry is not None:
                print(
                    f"▸ learnlog: refreshed  "
                    f"(corpus={entry['corpus_size']}, verified={entry['verified_count']})",
                    file=sys.stderr,
                )

    return judged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_input(path: Optional[str]) -> list[dict]:
    if path in (None, "-"):
        return json.load(sys.stdin)
    with open(path) as f:
        return json.load(f)


def _summarise(judged: list[dict]) -> None:
    """Print a short batch-level summary of judge-vs-generator agreement."""
    if not judged:
        return
    deltas = [r["delta"] for r in judged]
    gen_overalls = [r["generator_score"]["overall"] for r in judged]
    judge_overalls = [r["judge"]["overall"] for r in judged]
    print("─" * 50)
    print(f"batch summary  (n={len(judged)})")
    print(f"  generator mean overall: {statistics.fmean(gen_overalls):.2f}")
    print(f"  judge mean overall:     {statistics.fmean(judge_overalls):.2f}")
    print(f"  mean delta:             {statistics.fmean(deltas):+.2f}")
    if len(deltas) > 1:
        print(f"  delta stdev:            {statistics.stdev(deltas):.2f}")
    try:
        from settings import load_settings
        from llm import _resolve_models
        _, _, judge_model = _resolve_models(load_settings())
        print(f"  judge model:            {judge_model}")
    except Exception:
        pass


def _main() -> None:
    parser = argparse.ArgumentParser(description="Judge puns from pun_generator output.")
    parser.add_argument("--mode", choices=("llm", "human", "hybrid"), default="llm")
    parser.add_argument("--topic", required=True, help="topic the puns were generated about")
    parser.add_argument("--audience", required=True, help="audience the puns were aimed at")
    parser.add_argument("--tone", default="neutral", help="tone / register")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--input", "-i", help="path to generator JSON (or '-' for stdin)")
    src.add_argument(
        "--generate", "-g", type=int,
        help="generate this many puns first instead of reading input",
    )
    parser.add_argument("--format", choices=("json", "pretty"), default="pretty")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"min judge overall to enter corpus (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="don't write to corpus or refresh learnlog",
    )
    parser.add_argument(
        "--no-refresh", action="store_true",
        help="write to corpus but skip the learnlog refresh",
    )
    args = parser.parse_args()

    if args.generate:
        generator_output = generate_puns(
            args.generate, args.topic, args.audience, args.tone,
        )
    else:
        generator_output = _read_input(args.input)

    judged = judge_puns(
        generator_output, args.mode, args.topic, args.audience, args.tone,
        persist=not args.no_persist,
        threshold=args.threshold,
        refresh_learnlog=not args.no_refresh,
    )

    if args.format == "json":
        print(json.dumps(judged, indent=2))
        return

    for i, row in enumerate(judged, 1):
        p, g, j = row["pun"], row["generator_score"], row["judge"]
        print(f"{i}. {p['text']}")
        print(f"   pivot: {p['pivot']!r}  ~  {p['original']!r}")
        print(
            f"   gen overall: {g['overall']:.2f}    "
            f"judge overall: {j['overall']:.2f}    "
            f"Δ: {row['delta']:+.2f}    ({j['source']})"
        )
        if j.get("comment"):
            print(f"   comment: {j['comment']}")
    _summarise(judged)


if __name__ == "__main__":
    _main()
