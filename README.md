> [!IMPORTANT]  
> This project is programmed by _Claude Code_

# PUNGENERATOR

An AI pun generator, powered by general LLM models, which self-improves through phonetic scoring and LLM-as-judge systems.

## Metrics of judgement

> A pun is 'a humourous use of a word or phrase that has several meanings or that sounds like another word'.

A good pun is said to be when 'both meanings of the ambiguous word are true at the same time'. With this in mind, therefore the variables that measure the potency of a pun would be as follows:

- Phonetic distance: How close the 'anchor words' sound: processed with phonetic libraries
- Ambiguity: The presence of two similarly likely interpretations for a single sentence
- Distinctiveness: Both interpretations should be supported adequately by balanced sets of context words that support each meaning
- Surprise: The humour effect when a word appears unexpectedly in its local context but remains sensible within its global context

## Brief on the structure of the project

- **PUNGENERATOR**: A base pun generator, which takes in ① the number of puns to be generated in that batch, ② the topic of the puns, ③ the audiences targetted, and ④ the tone of the puns. Every time a pun is generated, learnt summaries are fed back into the system.
- **JUDGE mode**: The JUDGE mode is the functionality which provides self-improvement. It consists of the following modules:
  - **CORPUS**: When a pun is deemed a score higher than the threshold, it is stored into the CORPUS, tagged UNVERIFIED or VERIFIED based on respectively whether or not it was judged by another LLM model or alongside a user. 
  - **LEARNLOG**: At the end of each JUDGE session, a summary is appended to the LEARNLOG, taking input from the CORPUS and previous summaries to generate a guide for the system to follow.

## Files

| File              | Role                                                                 | AI-processed |
| ----------------- | -------------------------------------------------------------------- | ------------ |
| `CONDITIONS.txt`  | Plain-English brief on what counts as a good pun.                    | read-only    |
| `MODEL.txt`       | Hand-picked gold-standard puns. Strict pipe-delimited syntax.        | read-only    |
| `SETTINGS.txt`    | All tunable options (`key = value`, one per line).                   | read-only    |
| `corpus.json`     | Top puns, each marked VERIFIED / UNVERIFIED.                         | read/write   |
| `learnlog.json`   | History of AI-written "lessons" — latest one feeds the generator.    | written      |
| `archive.txt`     | JSON-lines dump of corpus entries demoted below threshold.           | written      |

For files intended wholly to be user-written, check the comments preceding their contents for information on the syntax.

## Data flow (A)

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

- Generator reads CONDITIONS, the latest learnlog summary, and a random sample of corpus exemplars (count from `SETTINGS.txt`).
- Judge reads CONDITIONS and rates puns against the same rubric; default judge model is different from the generator (set in `SETTINGS.txt`).
- Corpus tracks VERIFIED (human-touched) and UNVERIFIED (LLM-only) entries.
- Learnlog summary primarily compares MODEL.txt puns against the corpus to surface the gap between current output and the gold standard.

## Overload protection (A)

All API calls go through `safety.safe_complete`, which retries on:

- connection / timeout errors (both providers),
- rate limits (429),
- any 5xx (including Anthropic's 529 "Overloaded").

Backoff is exponential with jitter, capped at 60s, retry count from
`max_api_retries` in `SETTINGS.txt` (default 6).

## Installation

#### Installing the prerequisite libraries

```bash
pip3 install -r requirements.txt
```

#### LLM setup

The program works with either Anthropic or OpenAI. The former is recommended.

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

## Commands (A)

### One-liner (recommended)

```bash
python3 pun.py 5 "marine biology" "high-school teachers" --mode hybrid --tone playful
```

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


