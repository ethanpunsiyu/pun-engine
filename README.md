# PunGenerator

An AI pun generator that explicitly scores its output against the four
properties that make puns work (phonetic closeness, ambiguity,
distinctiveness, surprise) plus discourse relevance, judges the result with
an independent model (and optionally a human), and learns from a curated
corpus over time.

## Install

```bash
pip install -r requirements.txt
```

The codebase works with either Anthropic or OpenAI. Set the provider in
`SETTINGS.txt`:

```
provider = anthropic     # or: openai
generator_model =        # blank → use provider default
judge_model =            # blank → use provider default
```

Defaults if `generator_model` / `judge_model` are left blank:

| provider  | generator         | judge                |
| --------- | ----------------- | -------------------- |
| anthropic | `claude-opus-4-7` | `claude-sonnet-4-6`  |
| openai    | `gpt-5`           | `gpt-5-mini`         |

Export the matching API key:

```bash
export ANTHROPIC_API_KEY=...      # for provider = anthropic
export OPENAI_API_KEY=...         # for provider = openai
```

You only need to `pip install` the package for the provider you're using;
`requirements.txt` lists both for convenience.

## Files

| File              | Role                                                                 | AI-processed |
| ----------------- | -------------------------------------------------------------------- | ------------ |
| `CONDITIONS.txt`  | Plain-English brief on what counts as a good pun.                    | read-only    |
| `MODEL.txt`       | Hand-picked gold-standard puns. Strict pipe-delimited syntax.        | read-only    |
| `SETTINGS.txt`    | All tunable options (`key = value`, one per line).                   | read-only    |
| `corpus.json`     | Top puns, each marked VERIFIED / UNVERIFIED.                         | read/write   |
| `learnlog.json`   | History of AI-written "lessons" — latest one feeds the generator.    | written      |
| `archive.txt`     | JSON-lines dump of corpus entries demoted below threshold.           | written      |

`MODEL.txt`, `CONDITIONS.txt`, and `SETTINGS.txt` are meant to be edited by
hand. `corpus.json` is also safe to edit by hand if needed.

### MODEL.txt syntax

```
<pun text>
<pun text> | <rating 1-10>
<pun text> | <rating 1-10> | <comment>
```

`\|` inside the text escapes a literal pipe. Lines starting with `#` and
blank lines are ignored.

## Commands

### One-liner (recommended)

```bash
python3 pun.py 5 "marine biology" "high-school teachers" --mode hybrid --tone playful
```

Generates → judges → writes qualifying entries to the corpus → refreshes
the learnlog. Everything in one process, so human/hybrid prompts work
without stdin gymnastics.

Options:
- `--mode {llm, human, hybrid}` — judge mode (default `llm`).
- `--tone <string>` — register (`dry`, `playful`, `deadpan`, `formal`, …).
- `--threshold <float>` — overrides `SETTINGS.txt`.
- `--no-learnlog` — don't feed the latest lessons into the generator.
- `--no-persist` — judge in dry-run mode; nothing written to disk.
- `--no-refresh` — write to corpus but skip the learnlog refresh.
- `--format {pretty, json}` — output format.

### Per-step commands

#### Generator

```bash
python3 pun_generator.py 5 "tax law" "accountants" --tone dry
python3 pun_generator.py 5 "tax law" "accountants" --no-learnlog

# save the generated puns to a file for later judging:
python3 pun_generator.py 5 "tax law" "accountants" --format json > generated.json
```

#### Judge

```bash
# generate and judge in one step — works for all three modes:
python3 judge.py --mode llm    --topic "tax law" --audience "accountants" --tone dry -g 5
python3 judge.py --mode hybrid --topic "tax law" --audience "accountants" --tone dry -g 5
python3 judge.py --mode human  --topic "tax law" --audience "accountants" --tone dry -g 5 --no-persist

# (advanced) judge a previously-saved generator file instead:
python3 judge.py --mode llm --topic "tax law" --audience "accountants" -i generated.json

# save the judged output to disk for inspection or manual corpus import:
python3 judge.py --mode llm --topic "tax law" --audience "accountants" -g 5 \
  --no-persist --format json > judged.json
```

Flags: `--threshold`, `--no-persist`, `--no-refresh`, `--format`.

#### Corpus

```bash
python3 corpus.py stats
python3 corpus.py show --limit 20
python3 corpus.py show --verified-only
python3 corpus.py reconcile                  # apply SETTINGS.txt threshold
python3 corpus.py reconcile --threshold 0.7  # apply ad-hoc threshold

# manually import a judged JSON file (produced by `judge.py … --format json` above):
python3 corpus.py add --topic "tax law" --audience "accountants" -i judged.json
```

`reconcile` demotes any below-threshold entries to `archive.txt`.

#### Learnlog

```bash
python3 learnlog.py latest    # what gets fed into the generator
python3 learnlog.py refresh   # regenerate from current corpus + MODEL.txt
python3 learnlog.py history --limit 5
```

#### Settings (sanity check)

```bash
python3 settings.py           # prints loaded settings as JSON
```

## Data flow

```
            ┌── CONDITIONS.txt ─────────┐
            ▼                           ▼
   generator ──► judge ─┬─► corpus ─► archive.txt (below threshold)
       ▲                ├─► learnlog (refreshed when corpus changes)
       │                └─► returns judged rows
       │                                │
       │              ┌─ MODEL.txt ─────┤
       │              │                 ▼
       └─ latest_summary() ◄── learnlog summariser
```

- Generator reads CONDITIONS, the latest learnlog summary, and a random
  sample of corpus exemplars (count from `SETTINGS.txt`).
- Judge reads CONDITIONS and rates puns against the same rubric; default
  judge model is different from the generator (set in `SETTINGS.txt`).
- Corpus tracks VERIFIED (human-touched) and UNVERIFIED (LLM-only) entries.
- Learnlog summary primarily compares MODEL.txt puns against the corpus to
  surface the gap between current output and the gold standard.

## Overload protection

All API calls go through `safety.safe_complete`, which retries on:

- connection / timeout errors (both providers),
- rate limits (429),
- any 5xx (including Anthropic's 529 "Overloaded").

Backoff is exponential with jitter, capped at 60s, retry count from
`max_api_retries` in `SETTINGS.txt` (default 6).
