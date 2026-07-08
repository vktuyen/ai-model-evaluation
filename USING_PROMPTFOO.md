# Using promptfoo — a hands-on guide

A practical walkthrough so you can drive promptfoo yourself, not just run one script.

## 1. What it is (the mental model)

promptfoo is a **test runner for LLM prompts**. Think of it like unit tests, but for model output:

- In normal unit tests you write `input → assert(output == expected)`.
- With LLMs there's no single expected string, so instead you write `input → assert(output obeys these rules)`.

You give promptfoo two things — a **config file** (what to test) and a **command** (`promptfoo eval`). It sends every input to every model, checks each output against your rules, and shows you a scoreboard.

## 2. Why use it instead of a custom script

- One config tests **many models at once** and lays them side by side.
- Built-in **assertions** for the common checks (contains, regex, JSON-valid, "an LLM judge says it's good") — you don't reinvent them.
- **Latency and token usage** are recorded automatically.
- Results come with a **web UI** and CSV/JSON export.
- It **caches** model responses, so re-runs are fast and cheap.

You *could* write all this in Python, but you'd rebuild the scoreboard, the judge plumbing, caching, and reporting yourself.

## 3. Install

Nothing to install permanently — `npx` fetches it on demand:

```bash
npx promptfoo@latest eval
```

(Or install once globally: `npm install -g promptfoo`, then just `promptfoo eval`.) It needs Node.js 18+ (we use 20 on the server).

## 4. The smallest possible example

To really see how it works, here's a complete, self-contained config. Save it as `promptfooconfig.yaml` and run `npx promptfoo@latest eval`:

```yaml
prompts:
  - "Rephrase this professionally: {{text}}"

providers:
  - openai:gpt-4o-mini          # any model; needs that provider's API key

tests:
  - vars:
      text: "the boiler is broke innit"
    assert:
      - type: contains
        value: "boiler"
      - type: llm-rubric
        value: "sounds professional and keeps the meaning"
```

That's the whole idea: **prompts × providers × tests**, with **assertions** on each test. promptfoo runs every combination and scores it. Everything we built is just a bigger version of this.

## 5. Anatomy of a config

The four top-level pieces:

```yaml
prompts:      # the prompt template(s). {{var}} placeholders get filled from each test.
providers:    # the model(s) to test.
tests:        # the dataset: each row has vars (+ optional per-row asserts).
defaultTest:  # asserts/settings applied to EVERY test row.
```

**How we use each one** (`rephrase-eval/promptfooconfig.yaml`):

- **prompts** points at `prompt.js` — a small function that wraps each row into the real production prompt (system prompt + the `{"note":..,"tone":..,"type":..}` JSON). Using a function (instead of a plain string) lets us build the exact input the feature uses, and even tweak it per-model.
- **providers** are our three `llama.cpp` endpoints. Each has an `apiBaseUrl` (the server URL) and settings like `temperature` and `max_tokens`.
- **tests** = `tests.yaml`, our 38 rows. Each row supplies `note`, `tone`, `type`, and a `preserve` list.
- **defaultTest** holds the checks that run on every row (so we don't repeat them 38 times), plus the judge configuration.

## 6. Assertions — the part that matters most

Assertions are your rules. Two families:

**Deterministic (plain code, instant, free):**

```yaml
- type: contains          # output contains a string
  value: "boiler"
- type: regex             # output matches a pattern
  value: "\\d{2}/\\d{2}/\\d{4}"
- type: is-json           # output is valid JSON (we use the opposite idea)
- type: javascript        # your own JS check — the flexible one
  value: file://checks.js:preserveTokens
```

The `javascript` type is what powers our Layer 1 checks: `checks.js` exports functions (`preserveTokens`, `ukSpelling`, `plainTextNotJson`, `lengthRule`) that get the output + the test vars and return pass/fail with a reason.

**Model-graded (an LLM judges it):**

```yaml
- type: llm-rubric        # a judge model scores the output against a rubric
  value: "meaning preserved, correct tone, professional UK English"
```

This is our `semantic_quality` check. The judge model is set once in `defaultTest.options.provider` (Claude via OpenRouter).

**Speed:**

```yaml
- type: latency           # fails if a response is slower than the threshold
  threshold: 20000        # ms
```

Every assertion can carry a `metric:` label — that's how the report groups scores into columns like `preserve_details`, `uk_spelling`, etc. A test **passes only if all its assertions pass**.

## 7. The commands you'll actually use

```bash
# Run the eval
npx promptfoo@latest eval

# Point at a specific config
npx promptfoo@latest eval -c promptfooconfig.yaml

# Save machine-readable output
npx promptfoo@latest eval -o results.csv -o results.json

# One request at a time (clean latency; avoids overloading small models)
npx promptfoo@latest eval -j 1

# Ignore the cache and regenerate everything
npx promptfoo@latest eval --no-cache

# See per-request detail while it runs
npx promptfoo@latest eval --verbose

# Open the interactive side-by-side report
npx promptfoo@latest view
```

Our `run_on_server.sh` just runs `eval -j 1 --no-cache --verbose -o results.json -o results.csv` on the server and copies the results back.

> Tip: `promptfoo eval` **exits with a non-zero code when any test fails** — that's normal for an eval, but it's why our script appends `|| true` so it still pulls results back.

## 8. Reading the results

Three ways to look at the same run:

- **Terminal** — a model × test grid with pass/fail, plus a summary (X passed / Y failed) and token totals.
- **`promptfoo view`** — the best view: a table with one column per model, per-check pass rates, latency, and you can click any cell to see the exact input, output, and why each assertion passed or failed.
- **`results.csv`** — open in Excel/Sheets to pivot the numbers yourself.

## 9. The normal working loop

```
edit the prompt or a check  →  promptfoo eval  →  promptfoo view  →  see what failed  →  repeat
```

Because responses are cached, if you only change an assertion (not the prompt or model), the re-run reuses the cached outputs and just re-scores them — near-instant.

## 10. How our setup differs from the basic example

- The models are **remote**, so we run promptfoo **on the server** (via `run_on_server.sh`) where it can reach them.
- The judge (`llm-rubric`) runs on **Claude via OpenRouter**, needing `OPENROUTER_API_KEY`.
- The prompt is built by a **JS function** (`prompt.js`) instead of a plain string, so it matches production exactly.

Everything else is the plain promptfoo model from section 4 — just scaled up to 3 models, 38 tests, and 6 checks.

## Where to learn more

Official docs: https://promptfoo.dev/docs/ (the "Assertions & metrics" and "Configuration" pages are the most useful).
