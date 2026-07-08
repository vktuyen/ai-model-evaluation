# FSM AI Rephrase — Evaluation Harness

Evaluates the Joblogic AI Rephrase feature across the three locally-deployed
models (Gemma 4 E2B, Qwen3.5 2B, Llama 3.2 3B) for **accuracy** and **speed**,
using a reference-free approach: dataset = input + rules, no golden output.

## What it scores

Every model output is scored on two layers:

- **Deterministic checks** (`checks.js`, free/fast): required tokens preserved
  (numbers, dates, prices, job refs, names), UK spelling, plain-text-not-JSON,
  and the Make Shorter / Make Longer length rules.
- **Semantic rubric** (Claude as judge): meaning preserved, tone/type match,
  professional FSM phrasing, British English, no added info.

promptfoo also records **latency and token usage** per call automatically, so
you get speed stats for free.

## Files

- `promptfooconfig.yaml` — providers (3 endpoints), checks, judge config
- `tests.yaml` — the dataset (add rows here)
- `checks.js` — deterministic check functions
- `prompts/rephrase_system_prompt.txt` — the production system prompt
- `prompts/prompt.js` — wraps each row into the `{note,tone,type}` input

## Prerequisites

1. Node.js 18+ (`npx` runs promptfoo, no install needed).
2. An Anthropic API key for the judge:
   ```
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Network access from where you run this to the model ports.
   The config points at `http://10.30.11.110:8011|8002|8003`.

### Running against the SSH server

Either run the harness **on the server**, or open SSH tunnels from your
machine and point the config at localhost:

```bash
ssh -L 8011:localhost:8011 -L 8002:localhost:8002 -L 8003:localhost:8003 user@10.30.11.110
```

Then in `promptfooconfig.yaml` change the three `apiBaseUrl` values to
`http://localhost:8011/v1` etc. First confirm the endpoints answer:

```bash
curl http://10.30.11.110:8011/v1/models
```

## Run

```bash
cd rephrase-eval
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest view        # opens the side-by-side web UI
```

## Reading the results

The terminal prints a **model × test grid** with pass/fail per cell. The web
view (`view`) gives you the summary stats you want:

- **Pass rate per model** — overall and broken down by metric
  (`preserve_details`, `uk_spelling`, `output_format`, `length_rule`,
  `semantic_quality`).
- **Latency** — mean / p95 per model (the speed comparison).
- **Token usage** — per call.

Export for your own summary:

```bash
npx promptfoo@latest eval -c promptfooconfig.yaml -o results.csv
npx promptfoo@latest eval -c promptfooconfig.yaml -o results.json
```

`preserve_details` and `output_format` are the FSM-critical metrics — weight
them heavily; a wrong price or job number is a real bug even if the prose reads
well.

## Extending

- **More data:** add rows to `tests.yaml`. Put any value that must survive
  unchanged into that row's `preserve` list.
- **Cover all tones/types:** vary `tone` (Professional/Concise/Casual/Elaborate)
  and `type` (Default/Make Shorter/Make Longer) across rows.
- **Different judge:** change `defaultTest.options.provider` (e.g. a Claude
  Opus/Haiku model string your promptfoo version supports), or pass
  `--grader ...` on the CLI.
- **Repeatability:** add `-j 1` for sequential runs, or `--repeat 3` to average
  over multiple samples per case.
