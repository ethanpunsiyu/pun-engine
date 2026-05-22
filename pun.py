"""One-command pipeline: generate → judge (which now writes corpus + learnlog).

    python pun.py 5 "tax law" "accountants" --mode hybrid

Generation reads the latest learnlog summary as input; judging persists
qualifying puns to the corpus and refreshes the learnlog on the way out.
No separate corpus command needed in the normal flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from corpus import DEFAULT_THRESHOLD
from judge import judge_puns
from learnlog import latest_summary
from llm import LLMClient
from pun_generator import generate_puns
from settings import load_settings


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(
    n: int,
    topic: str,
    audience: str,
    tone: str = "neutral",
    *,
    mode: str = "llm",
    threshold: float = DEFAULT_THRESHOLD,
    use_learnlog: bool = True,
    persist: bool = True,
    refresh_learnlog: bool = True,
    client: Optional["LLMClient"] = None,
) -> list[dict]:
    if client is None:
        client = LLMClient()

    summary = latest_summary() if use_learnlog else None
    _log(
        f"▸ Generating {n} pun(s) about {topic!r} for {audience!r} (tone: {tone})"
        + (" with learnlog" if summary else "")
    )
    _log(
        f"  provider={client.provider}  "
        f"generator={client.generator_model}  "
        f"judge={client.judge_model}"
    )
    _log("  calling generator… (this can take 30-90s for reasoning models)")
    gen = generate_puns(
        n, topic, audience, tone,
        client=client, learnlog_summary=summary,
    )
    if not gen:
        _log("  no candidates produced")
        return []

    _log(f"▸ Judging ({mode})")
    return judge_puns(
        gen, mode, topic, audience, tone,
        client=client,
        persist=persist,
        threshold=threshold,
        refresh_learnlog=refresh_learnlog,
    )


def _print_pretty(judged: list[dict]) -> None:
    for i, row in enumerate(judged, 1):
        p, g, j = row["pun"], row["generator_score"], row["judge"]
        tag = "VERIFIED" if j["source"] in ("human", "hybrid") else "unverified"
        print(f"\n{i}. {p['text']}")
        print(f"   pivot: {p['pivot']!r} ~ {p['original']!r}")
        print(
            f"   gen: {g['overall']:.2f}   judge: {j['overall']:.2f}   "
            f"Δ: {row['delta']:+.2f}   [{tag}]"
        )
        if j.get("comment"):
            print(f"   comment: {j['comment']}")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, judge, store, and learn — in one command."
    )
    parser.add_argument("n", type=int, help="number of puns")
    parser.add_argument("topic", type=str)
    parser.add_argument("audience", type=str)
    parser.add_argument("--tone", default="neutral",
                        help="tone / register (e.g. dry, playful, deadpan, formal)")
    parser.add_argument("--mode", choices=("llm", "human", "hybrid"), default="llm")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="min judge overall to enter the corpus (defaults to SETTINGS.txt)",
    )
    parser.add_argument("--no-learnlog", action="store_true",
                        help="don't feed prior lessons into the generator")
    parser.add_argument("--no-persist", action="store_true",
                        help="don't write to corpus or refresh learnlog")
    parser.add_argument("--no-refresh", action="store_true",
                        help="write to corpus but skip the learnlog refresh")
    parser.add_argument("--format", choices=("json", "pretty"), default="pretty")
    args = parser.parse_args()

    threshold = args.threshold
    if threshold is None:
        try:
            threshold = float(load_settings()["threshold"])
        except Exception:
            threshold = DEFAULT_THRESHOLD

    judged = run(
        args.n, args.topic, args.audience, args.tone,
        mode=args.mode,
        threshold=threshold,
        use_learnlog=not args.no_learnlog,
        persist=not args.no_persist,
        refresh_learnlog=not args.no_refresh,
    )

    if args.format == "json":
        print(json.dumps(judged, indent=2))
    else:
        _print_pretty(judged)


if __name__ == "__main__":
    _main()
