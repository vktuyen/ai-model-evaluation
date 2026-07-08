# How the AI Model Evaluation Works

A plain-language guide to what we built and how it scores the models.

## The goal

We want to pick the best small model to run the **AI Rephrase** feature, judged on two things:

1. **Accuracy** — does it rephrase correctly (keeps the meaning, follows the FSM rules)?
2. **Speed** — how fast does it respond?

We're testing three models deployed on the GPU server: `gemma-4-e2b`, `qwen3.5-2b`, and `llama-3.2-3b`.

## The key idea: no "correct answer" needed

Rephrasing has no single right answer — there are many good ways to reword a sentence. So we can't compare the output to a fixed "golden" answer.

Instead, we check each output against **rules**. Our test data is simply:

> **input + the rules the output must obey**

If the output obeys the rules, it passes. This is called *reference-free* evaluation, and it's the standard way to score open-ended text tasks.

## Where everything runs

- **The models** live on the Linux GPU server. Each one is served by `llama.cpp` as an OpenAI-style API endpoint (one port per model).
- **The eval tool** is [promptfoo](https://promptfoo.dev). It runs *on the server* so it can reach the models directly, but we launch it from the laptop with one command that ships the project up over SSH and pulls the results back.
- **The judge** is Claude (via OpenRouter). It handles the checks that need real reading comprehension.

## Why we needed a GPU (CUDA)

At first the models ran on the server's **CPU** and were painfully slow — about **0.6 tokens/second**, so a single rephrase took minutes. The server has an NVIDIA GPU (RTX 4070 Ti SUPER, 16 GB), but nothing was using it.

**CUDA** is NVIDIA's toolkit that lets software run on the GPU. Two things were needed to switch the models onto the GPU:

1. **Build the model runtime with CUDA.** The runtime (`llama.cpp`) has to be compiled with the CUDA option turned on. That needs the CUDA toolkit installed (the `nvcc` compiler) — the graphics driver alone isn't enough.
2. **Launch each model with GPU offload (`-ngl`).** This tells `llama.cpp` to load the model's layers onto the GPU instead of the CPU.

After that, all three models sit on the GPU (~6 GB used, lots of headroom) and generation jumped from 0.6 t/s to tens of tokens per second — the reason a full run went from grinding for ages down to a few minutes.

The deploy script (`deploy_local_models_v5.py`) does both automatically: `--rebuild` compiles `llama.cpp` with CUDA, and it launches each model with `-ngl 99` (offload all layers). Running it with `--status` shows the GPU memory in use so you can confirm the models are really on the GPU.

## About promptfoo (the eval tool)

**promptfoo** is an open-source tool for testing and comparing LLM outputs. We chose it because it does exactly what we need with very little code:

- It talks to any **OpenAI-style API**, so it connects straight to our `llama.cpp` endpoints — no adapter needed.
- It runs the **same dataset against all three models** in one go and shows them side by side.
- It supports **both kinds of check** we need: code assertions (our Layer 1 checks) and an "LLM rubric" graded by another model (our Claude judge in Layer 2).
- It records **latency and token counts automatically** — that's our speed measurement for free.

Everything is configured in one file, `promptfooconfig.yaml`:

- **providers** — the three models (each endpoint URL and settings).
- **prompts** — how each input becomes a request (`prompt.js`).
- **tests** — the dataset file (`tests.yaml`).
- **defaultTest → assert** — the checks applied to every case (the code checks, the Claude rubric, and a latency guard).
- **defaultTest → options → provider** — the judge model (Claude via OpenRouter).

You run it with `promptfoo eval` (our `run_on_server.sh` does this on the server), and `promptfoo view` opens an interactive side-by-side web report.

## How one test case is scored

Every model output goes through two layers of checks.

### Layer 1 — Code checks (fast, free, objective)

Plain code (`checks.js`) verifies the things that have a definite right/wrong answer:

- **preserve_details** — every important value (job numbers, prices, dates, names) still appears, unchanged.
- **uk_spelling** — no US spellings (e.g. flags "organize" — should be "organise").
- **output_format** — the answer is plain text, not JSON, and doesn't leak the tone/type settings.
- **length_rule** — "Make Shorter" is actually shorter, "Make Longer" is actually longer.

### Layer 2 — LLM judge (Claude reads it)

Some things need judgement, so Claude grades:

- **semantic_quality** — meaning preserved, no invented details, correct tone, professional and readable.

### Plus: speed

promptfoo automatically records how long each response took. That's the **latency** number — our speed measure. No extra work needed.

A test case **passes overall only if every check passes.**

## Where the criteria come from

We did not invent these checks or borrow them from promptfoo. **Every criterion is extracted directly from the raw rephrase system prompt** — the exact instructions the remote API sends to the model in production (`prompts/rephrase_system_prompt.txt`). We read that prompt, pulled out each rule it states, and turned it into a measurable check. So the eval is literally testing whether each model obeys the feature's own written rules.

| Check | The rule it comes from (in the production system prompt) |
|---|---|
| preserve_details | "Maintain all names, job numbers, numerical values, dates (DD/MM/YYYY), and job-related references exactly as given" |
| uk_spelling | "Use UK business language and British grammar (e.g. 'categorise' instead of 'categorize')" |
| output_format | "Do not return the result in JSON format… The output should be plain text… Do not include the tone or type in your output" |
| length_rule | The Rephrase Type definitions: "Make Shorter" (condense) / "Make Longer" (expand) |
| semantic_quality | "preserving meaning… Do not introduce new information or make assumptions," plus the four tone definitions — graded by Claude |

Two numbers in the report are **not** from the prompt:

- **Median latency** comes from the project goal of measuring *speed*, not from the prompt. promptfoo records it automatically.
- **"Still dumps reasoning"** is not a defined check — it's a diagnostic observation we added by hand to explain why gemma/qwen failed.

### Rules in the prompt we don't check yet

The system prompt has two more rules we haven't turned into automated checks. They're good candidates to add:

- **Preserve structure** — "Maintain bullet points, lists, and structured formatting where applicable."
- **Keep hyperlinks unchanged** — "Improve structure where needed while keeping hyperlinks unchanged."

## The end-to-end flow

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

So `checks.js` sits at the **scoring** step: promptfoo takes each model's output and runs it through the `checks.js` functions (that's Layer 1) and, in parallel, sends it to Claude for the Layer 2 rubric. The config (`promptfooconfig.yaml`) is what wires the output into both — each `type: javascript` assertion points at a function in `checks.js`, and the `type: llm-rubric` assertion points at Claude.

You trigger the whole thing with one command:

```bash
./run_on_server.sh
```

## How to read the results

Open `results.csv` (or the web view). For each model you get a **pass rate per check** and the **median latency**. Compare models side by side:

- High deterministic scores + high semantic score = accurate.
- Low latency = fast.
- The `preserve_details` and `output_format` checks are the most important — a wrong price or a JSON blob is a real bug even if the sentence reads nicely.

## What we found so far

- **llama-3.2-3b is the clear winner** — clean plain-text output, passes 89–100% of the code checks, and responds in ~0.2s (10–20× faster than the others).
- **gemma and qwen dump their "thinking"** into the answer instead of just returning the rephrase, which fails the format and spelling checks and makes them much slower. They need reasoning turned off at the server before they're fair candidates.
- The eval is genuinely catching real quality issues (e.g. a model changing a date, adding words that weren't there, or not actually rephrasing), which is exactly what we want it to do.

## The files

| File | What it is |
|---|---|
| `tests.yaml` | The dataset: inputs + the values each must preserve |
| `checks.js` | Layer 1 code checks |
| `promptfooconfig.yaml` | Wires up the 3 models, the checks, and the Claude judge |
| `prompts/prompt.js` | Builds the production prompt for each input |
| `prompts/rephrase_system_prompt.txt` | The FSM rephrase system prompt |
| `run_on_server.sh` | Ships to the server, runs the eval, pulls results back |
| `results.csv` / `results.json` | The scored results |
| `deploy_local_models_v5.py` | Deploys the 3 models on the GPU (builds `llama.cpp` with CUDA, launches with `-ngl`) |
