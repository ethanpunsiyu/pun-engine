"""AI Pun Generator.

Generates a batch of puns for a given (topic, audience) and scores each one
on five dimensions derived from how puns actually work:

    phonetic        — how close the pivot word and its soundalike sound
                      (CMU pronouncing dict + double metaphone fallback)
    ambiguity       — both readings of the pivot are simultaneously plausible
    distinctiveness — each reading is anchored by its own balanced set of
                      surrounding context words
    surprise        — pivot is unexpected locally yet sensible globally
    relevance       — the utterance actually says something on-topic
                      for the discourse / audience

The composite score is a weighted sum and is used to rank an over-generated
candidate pool down to the `n` puns the caller asked for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional

import pronouncing
from metaphone import doublemetaphone

from llm import LLMClient
from safety import safe_complete


# Kept for backwards compatibility with any external code that imports it;
# the actual model in use comes from `LLMClient.generator_model`.
MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Pun:
    text: str                  # the pun, one utterance
    pivot: str                 # the ambiguous word as it appears in `text`
    original: str              # the word `pivot` sounds like / evokes
    meaning_local: str         # the literal reading of pivot in this sentence
    meaning_global: str        # the topic-aligned secondary reading
    context_local: list[str]   # words anchoring meaning_local
    context_global: list[str]  # words anchoring meaning_global


@dataclass
class PunScore:
    phonetic: float        # 1 = identical pronunciation, 0 = nothing alike
    ambiguity: float       # 0..1
    distinctiveness: float # 0..1
    surprise: float        # 0..1
    relevance: float       # 0..1
    overall: float         # weighted composite, 0..1


WEIGHTS = {
    "phonetic": 0.20,
    "ambiguity": 0.25,
    "distinctiveness": 0.15,
    "surprise": 0.20,
    "relevance": 0.20,
}


# ---------------------------------------------------------------------------
# Phonetic distance — rule-based, using phonetic libraries
# ---------------------------------------------------------------------------


def _arpabet(word: str) -> list[str]:
    """Return the CMU ARPAbet phoneme sequence for `word` (no stress)."""
    entries = pronouncing.phones_for_word(word.lower().strip())
    if not entries:
        return []
    return [re.sub(r"\d", "", p) for p in entries[0].split()]


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def phonetic_similarity(a: str, b: str) -> float:
    """Phonetic similarity between two words, 0 (different) .. 1 (identical).

    Uses CMU ARPAbet edit distance when both words are in the dictionary,
    otherwise falls back to double-metaphone code agreement.
    """
    if not a or not b:
        return 0.0
    if a.lower() == b.lower():
        return 1.0
    pa, pb = _arpabet(a), _arpabet(b)
    if pa and pb:
        m = max(len(pa), len(pb))
        return 1.0 - (_levenshtein(pa, pb) / m if m else 0.0)
    ma, mb = doublemetaphone(a), doublemetaphone(b)
    codes_a = {c for c in ma if c}
    codes_b = {c for c in mb if c}
    if not codes_a or not codes_b:
        return 0.0
    return len(codes_a & codes_b) / max(len(codes_a), len(codes_b))


# ---------------------------------------------------------------------------
# Semantic dimensions — LLM-rated against a strict rubric
# ---------------------------------------------------------------------------


SCORING_SYSTEM = """You are a careful pun analyst rating puns against a fixed rubric.

For EACH pun in the input, produce one entry. Return ALL entries wrapped in
a single JSON object with shape:

{
  "rows": [
    {
      "id": <int>,
      "ambiguity": <float 0..1>,
      "distinctiveness": <float 0..1>,
      "surprise": <float 0..1>,
      "relevance": <float 0..1>,
      "notes": "<one short sentence explaining the lowest score>"
    },
    ...
  ]
}

Rubric (be strict — mediocre puns sit around 0.4-0.6):

- ambiguity:        how plausibly BOTH meanings of the pivot fit the sentence at the
                    same time. 0 = only one reading is real; 1 = both equally valid.
- distinctiveness:  do the context words anchor each meaning separately and in a
                    balanced way? 0 = one meaning is unsupported, or both lean on the
                    same evidence; 1 = each meaning has its own clear context cues.
- surprise:         is the pivot unexpected at its local position but still sensible
                    inside the global topic? 0 = obvious or jarringly wrong;
                    1 = the second reading lands as a small reveal.
- relevance:        does the utterance carry on-topic content for the audience, or
                    is it empty / disconnected? 0 = off-topic filler; 1 = says
                    something coherent about the topic.

Return ONLY the JSON object. No prose, no code fences.
"""


GEN_SYSTEM = """You are a pun generator. Given a topic, audience, tone, and a count, produce that many original puns.

Each pun must:
  1. Be a complete utterance that makes sense as discourse — it should say
     something coherent and on-topic, not a non-sequitur.
  2. Contain a pivot word that supports two readings simultaneously: a literal
     local reading and a topic-aligned reading.
  3. Have distinguishable context cues for each reading in the surrounding words.
  4. Use a pivot that sounds like (or is identical to) a word that anchors the
     topic-aligned reading.
  5. Be tuned to the audience's vocabulary and frame of reference.
  6. Match the requested tone in register and rhythm, NOT in content. Tone
     shapes how it sounds; the five mechanics above stay constant.

Return ONLY valid JSON with this exact shape (no prose, no code fences):
{
  "puns": [
    {
      "text": "<the pun, one sentence>",
      "pivot": "<the ambiguous word as it appears in text>",
      "original": "<the word it sounds like / evokes>",
      "meaning_local": "<the literal local reading>",
      "meaning_global": "<the topic-aligned reading>",
      "context_local": ["<word>", ...],
      "context_global": ["<word>", ...]
    }
  ]
}
"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def generate_candidates(
    client: "LLMClient",
    n: int,
    topic: str,
    audience: str,
    tone: str = "neutral",
    over_factor: int = 2,
    learnlog_summary: Optional[str] = None,
    exemplars: Optional[list[dict]] = None,
    conditions: Optional[str] = None,
) -> list[Pun]:
    """Over-generate puns so ranking has something to choose from.

    System prompt is layered for caching:
        block 1 (cached): static GEN_SYSTEM
        block 2 (cached): CONDITIONS.txt — changes rarely
        block 3 (cached): learnlog summary — changes per corpus update
    Per-call data (topic, audience, tone, exemplars) goes in the user message.
    """
    request_n = max(n * over_factor, n + 2)

    system_blocks = [{"type": "text", "text": GEN_SYSTEM, "cache_control": {"type": "ephemeral"}}]
    if conditions:
        system_blocks.append({
            "type": "text",
            "text": "Conditions (what counts as a good pun):\n\n" + conditions,
            "cache_control": {"type": "ephemeral"},
        })
    if learnlog_summary:
        system_blocks.append({
            "type": "text",
            "text": "Lessons learned from prior batches (apply these):\n\n" + learnlog_summary,
            "cache_control": {"type": "ephemeral"},
        })

    user_lines = [
        f"Topic: {topic}",
        f"Audience: {audience}",
        f"Tone: {tone}",
        f"Count: {request_n}",
    ]
    if exemplars:
        user_lines.append("")
        user_lines.append("Exemplars from prior batches (for reference — do not copy):")
        user_lines.append(json.dumps(exemplars, indent=2))

    text = safe_complete(
        client,
        model=client.generator_model,
        max_tokens=4096,
        system=system_blocks,
        user="\n".join(user_lines),
    )
    data = json.loads(_strip_fences(text))
    return [Pun(**item) for item in data["puns"]]


def score_semantic_batch(
    client: "LLMClient",
    puns: list[Pun],
    topic: str,
    audience: str,
) -> list[dict]:
    """One LLM round-trip for the whole batch — cheaper and keeps relative
    scoring consistent."""
    payload = {
        "topic": topic,
        "audience": audience,
        "puns": [{"id": i, **asdict(p)} for i, p in enumerate(puns)],
    }
    text = safe_complete(
        client,
        model=client.generator_model,
        max_tokens=2048,
        system=[{"type": "text", "text": SCORING_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        user=json.dumps(payload, indent=2),
    )
    parsed = json.loads(_strip_fences(text))
    rows = parsed["rows"] if isinstance(parsed, dict) and "rows" in parsed else parsed
    by_id = {row["id"]: row for row in rows}
    # Preserve original order, fill gaps with neutral 0.5 if the model missed one.
    return [
        by_id.get(i, {"ambiguity": 0.5, "distinctiveness": 0.5, "surprise": 0.5, "relevance": 0.5})
        for i in range(len(puns))
    ]


def score_pun(pun: Pun, semantic: dict) -> PunScore:
    parts = {
        "phonetic": phonetic_similarity(pun.pivot, pun.original),
        "ambiguity": float(semantic["ambiguity"]),
        "distinctiveness": float(semantic["distinctiveness"]),
        "surprise": float(semantic["surprise"]),
        "relevance": float(semantic["relevance"]),
    }
    overall = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS)
    return PunScore(**parts, overall=overall)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_puns(
    n: int,
    topic: str,
    audience: str,
    tone: str = "neutral",
    *,
    client: Optional["LLMClient"] = None,
    learnlog_summary: Optional[str] = None,
    exemplars: Optional[list[dict]] = None,
    conditions: Optional[str] = None,
) -> list[dict]:
    """Generate `n` puns about `topic` for `audience` in `tone`, ranked by
    composite score.

    Unset optional args are auto-loaded:
      - `conditions` from CONDITIONS.txt
      - `exemplars` sampled from the corpus (count from SETTINGS.txt)
      - `learnlog_summary` from the latest learnlog entry

    Returns a list of dicts: {"pun": {...}, "score": {...}}, sorted high to low.
    """
    if n <= 0:
        return []
    if client is None:
        client = LLMClient()

    # Auto-load context bits if the caller didn't supply them.
    if conditions is None:
        from conditions import load_conditions
        conditions = load_conditions()
    if exemplars is None:
        try:
            from settings import load_settings
            from corpus import load_corpus, sample_exemplars
            k = int(load_settings()["exemplar_count"])
            exemplars = sample_exemplars(load_corpus(), k)
        except Exception:
            exemplars = []
    if learnlog_summary is None:
        try:
            from learnlog import latest_summary
            learnlog_summary = latest_summary()
        except Exception:
            learnlog_summary = None

    over_factor = 2
    try:
        from settings import load_settings
        over_factor = int(load_settings()["over_factor"])
    except Exception:
        pass

    candidates = generate_candidates(
        client, n, topic, audience, tone,
        over_factor=over_factor,
        learnlog_summary=learnlog_summary,
        exemplars=exemplars,
        conditions=conditions,
    )
    if not candidates:
        return []
    semantics = score_semantic_batch(client, candidates, topic, audience)
    scored = [
        {"pun": asdict(c), "score": asdict(score_pun(c, s))}
        for c, s in zip(candidates, semantics)
    ]
    scored.sort(key=lambda r: r["score"]["overall"], reverse=True)
    return scored[:n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of ranked puns.")
    parser.add_argument("n", type=int, help="number of puns to return")
    parser.add_argument("topic", type=str, help="subject the puns should be about")
    parser.add_argument("audience", type=str, help="who the puns are for")
    parser.add_argument("--tone", default="neutral",
                        help="tone / register (e.g. dry, playful, deadpan, formal)")
    parser.add_argument(
        "--format",
        choices=("json", "pretty"),
        default="pretty",
        help="output format (default: pretty)",
    )
    parser.add_argument(
        "--no-learnlog",
        action="store_true",
        help="don't load the latest learnlog summary into the generator prompt",
    )
    args = parser.parse_args()

    summary = None
    if not args.no_learnlog:
        try:
            from learnlog import latest_summary
            summary = latest_summary()
        except Exception:
            summary = None

    results = generate_puns(
        args.n, args.topic, args.audience, args.tone,
        learnlog_summary=summary,
    )

    if args.format == "json":
        print(json.dumps(results, indent=2))
        return

    for i, row in enumerate(results, 1):
        p, s = row["pun"], row["score"]
        print(f"{i}. {p['text']}")
        print(f"   pivot: {p['pivot']!r} ~ {p['original']!r}")
        print(
            f"   scores: overall={s['overall']:.2f}  "
            f"phon={s['phonetic']:.2f}  amb={s['ambiguity']:.2f}  "
            f"dist={s['distinctiveness']:.2f}  surp={s['surprise']:.2f}  "
            f"rel={s['relevance']:.2f}"
        )
        print()


if __name__ == "__main__":
    _main()
