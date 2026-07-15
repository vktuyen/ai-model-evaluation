# AI Model Evaluation

Evaluating small local LLMs for Joblogic's AI features (Field Service Management) on **accuracy** and **speed**. Each model is deployed on a GPU Linux server behind an **OpenAI-compatible endpoint**, on the **same runtime + quantization the Persona mobile app uses on-device**, then scored with [promptfoo](https://promptfoo.dev) (reference-free: dataset = input + rules).

Why runtime parity matters: same model *name* ≠ same behaviour. The eval only predicts on-device quality if it runs the model the way the phone does.

| Model | Port | Runtime (server = mobile) | Quantization |
|---|---|---|---|
| `gemma-4-e2b` | 8011 | **LiteRT-LM** (`litert-lm serve`) — CPU per bundle (= app's Android path) | `.litertlm` |
| `qwen3.5-2b` | 8002 | **llama.cpp** (`llama-server`) — matches the app's `fllama` | GGUF Q4_K_M |
| `llama-3.2-3b-instruct` | 8003 | **llama.cpp** (`llama-server`) | GGUF Q4_K_M |

`deploy_local_models_v6.py` deploys all three (both runtimes) with one command.

---

## Quick start

```bash
# 1. From your laptop: copy the deploy script up and start all 3 endpoints.
scp deploy_local_models_v6.py ml-server:~/
ssh ml-server "python3 ~/deploy_local_models_v6.py --restart --wait-minutes 30"
#   ^ first time on a fresh box, also add --rebuild (builds llama.cpp w/ CUDA)

# 2. Run the eval (ships to the server, runs promptfoo there, builds the report).
cd rephrase-eval
./run_on_server.sh
./pull_report.sh
```

That's the whole loop. The two sections below list the useful commands for each shell.

---

## Commands — on your laptop

```bash
# Ship the deploy script to the server
scp deploy_local_models_v6.py ml-server:~/

# Deploy / control the endpoints remotely (one-liners over SSH)
ssh ml-server "python3 ~/deploy_local_models_v6.py --status"
ssh ml-server "python3 ~/deploy_local_models_v6.py --restart --wait-minutes 30"
ssh ml-server "python3 ~/deploy_local_models_v6.py --stop"

# Run the evaluation (from the rephrase-eval/ folder)
cd rephrase-eval
./run_on_server.sh          # ship + run promptfoo on the server
./pull_report.sh            # pull results, build report.html, archive a snapshot

# Open a shell on the server
ssh ml-server

# Tunnel the endpoints to localhost (if you want to hit them from the laptop)
ssh -L 8011:localhost:8011 -L 8002:localhost:8002 -L 8003:localhost:8003 ml-server
```

`ml-server` is the SSH alias in `~/.ssh/config` (→ `tuyenv@10.30.11.110`).

---

## Commands — on the server shell

Run these after `ssh ml-server`.

```bash
# ---- Deploy script (handles both runtimes) --------------------------------
python3 ~/deploy_local_models_v6.py --restart --wait-minutes 30     # all 3
python3 ~/deploy_local_models_v6.py --restart --models gemma        # just Gemma (LiteRT-LM)
python3 ~/deploy_local_models_v6.py --restart --models qwen,llama   # just llama.cpp models
python3 ~/deploy_local_models_v6.py --status                        # ports + endpoints
python3 ~/deploy_local_models_v6.py --stop                          # stop all
python3 ~/deploy_local_models_v6.py --restart --rebuild             # rebuild llama.cpp (CUDA)
python3 ~/deploy_local_models_v6.py --restart --models gemma --reimport    # re-fetch .litertlm
python3 ~/deploy_local_models_v6.py --restart --models qwen --redownload   # re-fetch GGUF
python3 ~/deploy_local_models_v6.py --restart --models qwen,llama --gpu-layers 0  # CPU (fair speed)

# ---- Inspect ---------------------------------------------------------------
tmux ls                                             # gemma-litertlm / qwen35-2b / llama32-3b
tail -F ~/local_model_server/logs/*.log             # live logs
for p in 8011 8002 8003; do curl -sf http://127.0.0.1:$p/v1/models >/dev/null && echo "$p OK" || echo "$p DOWN"; done
~/litert-venv/bin/litert-lm list                    # imported LiteRT-LM models
nvidia-smi                                           # GPU usage

# ---- Smoke-test each endpoint ---------------------------------------------
# Gemma (LiteRT-LM). First request is a cold load (~40s); later ones are fast.
curl -s http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e2b","messages":[{"role":"user","content":"Rephrase professionally: job done ok"}],"temperature":0,"max_tokens":60}'

# Qwen (llama.cpp) — MUST send enable_thinking:false, else it burns tokens on
# hidden reasoning and returns empty text (Qwen3.5 is a thinking model).
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-2b","messages":[{"role":"user","content":"Rephrase professionally: job done ok"}],"temperature":0,"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}'

# Llama (llama.cpp)
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"Rephrase professionally: job done ok"}],"temperature":0,"max_tokens":60}'
```

### Manual deploy (what the script automates)

<details>
<summary>Gemma via LiteRT-LM (port 8011)</summary>

```bash
# Install the CLI into a venv (once). Needs Python 3.10+.
python3 -m venv ~/litert-venv && ~/litert-venv/bin/pip install --upgrade litert-lm
LB=~/litert-venv/bin/litert-lm

# Import the SAME bundle the app downloads, under the id the eval expects.
# import --from-huggingface-repo <repo> <FILE.litertlm> <MODEL_ID>
$LB import --from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm gemma-4-e2b

# Serve. NOTE: `serve` takes NO model id and NO backend flag — it hosts all
# imported models, routing by the request's `model` field, on the backend baked
# into the bundle (CPU for this Gemma = the app's Android path).
mkdir -p ~/local_model_server/logs
tmux new-session -d -s gemma-litertlm \
  "$LB serve --host 0.0.0.0 --port 8011 --verbose 2>&1 | tee -a ~/local_model_server/logs/gemma-litertlm.log"
```
</details>

<details>
<summary>Qwen + Llama via llama.cpp (ports 8002, 8003)</summary>

```bash
cd ~/local_model_server
# Build llama.cpp with CUDA (once)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git      # if missing
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --target llama-server

# Download the GGUFs (once)
mkdir -p models/qwen35_2b models/llama32_3b
curl -L --fail -o models/qwen35_2b/Qwen3.5-2B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf
curl -L --fail -o models/llama32_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Serve each (-ngl 99 = all layers on GPU; 0 = CPU; --jinja = model's chat template)
SRV=~/local_model_server/llama.cpp/build/bin/llama-server
tmux new-session -d -s qwen35-2b  "$SRV -m models/qwen35_2b/Qwen3.5-2B-Q4_K_M.gguf        --host 0.0.0.0 --port 8002 -c 4096 --alias qwen3.5-2b            -ngl 99 --jinja 2>&1 | tee -a logs/qwen35-2b.log"
tmux new-session -d -s llama32-3b "$SRV -m models/llama32_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8003 -c 4096 --alias llama-3.2-3b-instruct -ngl 99 --jinja 2>&1 | tee -a logs/llama32-3b.log"
```
</details>

---

## Gotchas

- **`litert-lm serve` has no backend/model flag.** Backend is baked into the `.litertlm` bundle; the Gemma bundle is CPU-constrained — which *is* the app's Android path (its WebGPU/Dawn GPU path fails on Android). CPU here is the faithful match, not a limitation.
- **Cold load:** the first LiteRT-LM request loads ~2.4 GB (~40 s). It stays resident after. Don't mistake it for a hang.
- **Qwen thinking:** always send `chat_template_kwargs.enable_thinking=false` (the eval config does). Otherwise the visible answer can be empty.
- **Speed comparison:** Gemma is CPU (= Android), Qwen/Llama default to GPU. For a fair on-device *latency* comparison, run Qwen/Llama with `--gpu-layers 0`. Quality parity is unaffected.

## Layout

```
deploy_local_models_v6.py   One deploy script — LiteRT-LM (Gemma) + llama.cpp (Qwen/Llama)
rephrase-eval/              The AI Rephrase evaluation
  promptfooconfig.yaml      Wires up the 3 endpoints, checks, and the Claude judge
  tests.yaml                Dataset: inputs + values each must preserve
  checks.js                 Deterministic checks (Layer 1)
  prompts/                  The production system prompt + prompt builder
  run_on_server.sh          Ship to server + run the eval
  pull_report.sh            Pull results + build/open the HTML report
  make_report.py            Builds the self-contained report.html
```

## Notes

Secrets (`.env.local`, e.g. the OpenRouter judge key) and generated output (`results.*`, `report.html`, `reports/`) are gitignored and never committed.

---

# How the evaluation works

A plain-language guide to what we built and how it scores the models.

### The goal

We want to pick the best small model to run the **AI Rephrase** feature, judged on two things:

1. **Accuracy** — does it rephrase correctly (keeps the meaning, follows the FSM rules)?
2. **Speed** — how fast does it respond?

We're testing three models: `gemma-4-e2b`, `qwen3.5-2b`, and `llama-3.2-3b`.

### The key idea: no "correct answer" needed

Rephrasing has no single right answer — there are many good ways to reword a sentence. So we can't compare the output to a fixed "golden" answer.

Instead, we check each output against **rules**. Our test data is simply:

> **input + the rules the output must obey**

If the output obeys the rules, it passes. This is called *reference-free* evaluation, and it's the standard way to score open-ended text tasks.

### Where everything runs

- **The models** live on the Linux GPU server, each behind an OpenAI-style API endpoint (one port per model) — on the runtime that matches the mobile app (LiteRT-LM for Gemma, `llama.cpp` for Qwen/Llama; see the table at the top).
- **The eval tool** is [promptfoo](https://promptfoo.dev). It runs *on the server* so it can reach the models directly, but we launch it from the laptop with one command that ships the project up over SSH and pulls the results back.
- **The judge** is Claude (via OpenRouter). It handles the checks that need real reading comprehension.

### Why we needed a GPU (CUDA) for the llama.cpp models

At first the models ran on the server's **CPU** and were painfully slow — about **0.6 tokens/second**, so a single rephrase took minutes. The server has an NVIDIA GPU (RTX 4070 Ti SUPER, 16 GB), but nothing was using it.

**CUDA** is NVIDIA's toolkit that lets software run on the GPU. Two things were needed to switch the llama.cpp models onto the GPU:

1. **Build the runtime with CUDA.** `llama.cpp` has to be compiled with the CUDA option on. That needs the CUDA toolkit installed (the `nvcc` compiler) — the graphics driver alone isn't enough.
2. **Launch each model with GPU offload (`-ngl`).** This loads the model's layers onto the GPU instead of the CPU.

After that, Qwen/Llama sit on the GPU and generation jumped from 0.6 t/s to tens of tokens per second. `deploy_local_models_v6.py --rebuild` compiles `llama.cpp` with CUDA and launches with `-ngl 99`; `--status` shows GPU memory in use. (Gemma runs on **CPU** via LiteRT-LM by design — that's the app's Android path; see Gotchas.)

### About promptfoo (the eval tool)

**promptfoo** is an open-source tool for testing and comparing LLM outputs. We chose it because it does exactly what we need with very little code:

- It talks to any **OpenAI-style API**, so it connects straight to our endpoints — no adapter needed.
- It runs the **same dataset against all three models** in one go and shows them side by side.
- It supports **both kinds of check** we need: code assertions (Layer 1) and an "LLM rubric" graded by another model (our Claude judge in Layer 2).
- It records **latency and token counts automatically** — that's our speed measurement for free.

Everything is configured in `rephrase-eval/promptfooconfig.yaml`:

- **providers** — the three endpoints (each URL and settings).
- **prompts** — how each input becomes a request (`prompt.js`).
- **tests** — the dataset file (`tests.yaml`).
- **defaultTest → assert** — the checks applied to every case (code checks, the Claude rubric, a latency guard).
- **defaultTest → options → provider** — the judge model (Claude via OpenRouter).

### How one test case is scored

Every model output goes through two layers of checks.

**Layer 1 — Code checks (fast, free, objective)** — plain code (`checks.js`) verifies things with a definite right/wrong answer:

- **preserve_details** — every important value (job numbers, prices, dates, names) still appears, unchanged.
- **uk_spelling** — no US spellings (flags "organize" — should be "organise").
- **output_format** — plain text, not JSON, and doesn't leak the tone/type settings.
- **length_rule** — "Make Shorter" is shorter, "Make Longer" is longer.

**Layer 2 — LLM judge (Claude reads it)** — for things that need judgement:

- **semantic_quality** — meaning preserved, no invented details, correct tone, professional and readable.

**Plus speed** — promptfoo records how long each response took (the **latency** number). No extra work.

A test case **passes overall only if every check passes.**

### Where the criteria come from

We did not invent these checks. **Every criterion is extracted directly from the raw rephrase system prompt** — the exact instructions the remote API sends to the model in production (`prompts/rephrase_system_prompt.txt`).

| Check | The rule it comes from (in the production system prompt) |
|---|---|
| preserve_details | "Maintain all names, job numbers, numerical values, dates (DD/MM/YYYY), and job-related references exactly as given" |
| uk_spelling | "Use UK business language and British grammar (e.g. 'categorise' instead of 'categorize')" |
| output_format | "Do not return the result in JSON format… The output should be plain text… Do not include the tone or type in your output" |
| length_rule | The Rephrase Type definitions: "Make Shorter" (condense) / "Make Longer" (expand) |
| semantic_quality | "preserving meaning… Do not introduce new information or make assumptions," plus the four tone definitions — graded by Claude |

Two numbers in the report are **not** from the prompt: **median latency** (the speed goal; promptfoo records it) and **"Still dumps reasoning"** (a hand-added diagnostic explaining why gemma/qwen failed).

Rules in the prompt we don't check yet (good candidates to add): **preserve structure** (bullets/lists) and **keep hyperlinks unchanged**.

### The end-to-end flow

```
tests.yaml         →  the dataset (input + rules)
prompt.js          →  wraps each input into the exact production prompt
   ↓ (sent to each model on the server)
model output       →  the raw text the model returned
   ↓ (promptfoo scores that output two ways)
   ├─ checks.js     →  Layer 1: code checks (preserve, UK spelling, format, length)
   └─ Claude judge  →  Layer 2: semantic_quality rubric
   ↓ (a case passes only if every check passes)
results.csv/json   →  pulled back to the laptop
```

### How to read the results

Open `results.csv` (or `promptfoo view`). For each model you get a **pass rate per check** and the **median latency**:

- High deterministic scores + high semantic score = accurate. Low latency = fast.
- `preserve_details` and `output_format` are the most important — a wrong price or a JSON blob is a real bug even if the sentence reads nicely.

### What we found so far

- **llama-3.2-3b was the clear winner** in early runs — clean plain-text output, 89–100% of the code checks, ~0.2s responses.
- **gemma and qwen dumped their "thinking"** into the answer, failing format/spelling checks and running slower. They need reasoning turned off at the server (Qwen: `enable_thinking:false`; see Gotchas) before they're fair candidates.
- The eval is genuinely catching real quality issues (changed dates, invented words, non-rephrases) — exactly what we want.

> Note: these findings predate the LiteRT-LM Gemma cutover. Re-run the eval to get numbers that reflect the on-device runtimes.

---

# Using promptfoo — a hands-on guide

A practical walkthrough so you can drive promptfoo yourself, not just run one script.

### The mental model

promptfoo is a **test runner for LLM prompts** — like unit tests, but for model output. Normal tests do `input → assert(output == expected)`; with LLMs there's no single expected string, so you do `input → assert(output obeys these rules)`. You give it a **config file** (what to test) and a **command** (`promptfoo eval`); it sends every input to every model, checks each output against your rules, and shows a scoreboard.

### Why use it instead of a custom script

- One config tests **many models at once**, side by side.
- Built-in **assertions** (contains, regex, JSON-valid, "an LLM judge says it's good").
- **Latency and token usage** recorded automatically.
- **Web UI** + CSV/JSON export, and it **caches** responses so re-runs are fast.

### Install

Nothing permanent — `npx` fetches it on demand (needs Node.js 18+; we use 20 on the server):

```bash
npx promptfoo@latest eval
```

### The smallest possible example

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

The whole idea: **prompts × providers × tests**, with **assertions** on each test. Everything we built is a bigger version of this.

### Anatomy of our config (`rephrase-eval/promptfooconfig.yaml`)

```yaml
prompts:      # the prompt template(s). {{var}} placeholders get filled from each test.
providers:    # the model(s) to test.
tests:        # the dataset: each row has vars (+ optional per-row asserts).
defaultTest:  # asserts/settings applied to EVERY test row.
```

- **prompts** points at `prompt.js` — wraps each row into the real production prompt (system prompt + the `{"note":..,"tone":..,"type":..}` JSON), so the input matches production exactly.
- **providers** are our three endpoints — each with an `apiBaseUrl` and settings like `temperature`/`max_tokens`.
- **tests** = `tests.yaml` (our ~38 rows: `note`, `tone`, `type`, and a `preserve` list).
- **defaultTest** holds the checks that run on every row, plus the judge config.

### Assertions — the part that matters most

**Deterministic (plain code, instant, free):**

```yaml
- type: contains
  value: "boiler"
- type: regex
  value: "\\d{2}/\\d{2}/\\d{4}"
- type: is-json          # output is valid JSON (we use the opposite idea)
- type: javascript       # your own JS check — the flexible one
  value: file://checks.js:preserveTokens
```

The `javascript` type powers our Layer 1 checks (`checks.js` exports `preserveTokens`, `ukSpelling`, `plainTextNotJson`, `lengthRule`).

**Model-graded (an LLM judges it):**

```yaml
- type: llm-rubric       # a judge model scores the output against a rubric
  value: "meaning preserved, correct tone, professional UK English"
```

This is our `semantic_quality` check; the judge is set once in `defaultTest.options.provider` (Claude via OpenRouter).

**Speed:**

```yaml
- type: latency
  threshold: 20000       # ms
```

Every assertion can carry a `metric:` label — that's how the report groups scores into columns (`preserve_details`, `uk_spelling`, …). A test **passes only if all its assertions pass**.

### The commands you'll actually use

```bash
npx promptfoo@latest eval                        # run the eval
npx promptfoo@latest eval -c promptfooconfig.yaml  # specific config
npx promptfoo@latest eval -o results.csv -o results.json  # save output
npx promptfoo@latest eval -j 1                   # one request at a time (clean latency)
npx promptfoo@latest eval --no-cache             # ignore cache, regenerate
npx promptfoo@latest eval --verbose              # per-request detail
npx promptfoo@latest view                        # interactive side-by-side report
```

Our `run_on_server.sh` runs `eval -j 1 --no-cache --verbose -o results.json -o results.csv` on the server and copies results back.

> Tip: `promptfoo eval` **exits non-zero when any test fails** — normal for an eval, which is why the script appends `|| true` so it still pulls results back.

### Reading the results

- **Terminal** — a model × test grid with pass/fail, a summary, and token totals.
- **`promptfoo view`** — the best view: one column per model, per-check pass rates, latency; click any cell for the exact input/output and why each assertion passed/failed.
- **`results.csv`** — open in Excel/Sheets to pivot yourself.

### The normal working loop

```
edit the prompt or a check  →  promptfoo eval  →  promptfoo view  →  see what failed  →  repeat
```

Because responses are cached, changing only an assertion (not the prompt/model) re-scores cached outputs — near-instant.

Official docs: https://promptfoo.dev/docs/ (the "Assertions & metrics" and "Configuration" pages are the most useful).
