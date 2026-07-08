# AI Model Evaluation

Evaluating small local LLMs for Joblogic's AI features (Field Service Management) on **accuracy** and **speed**. The models are deployed on a GPU Linux server via `llama.cpp` (OpenAI-compatible endpoints) and scored with [promptfoo](https://promptfoo.dev) using a reference-free approach: dataset = input + rules.

## Documentation

- **[HOW_THE_EVALUATION_WORKS.md](HOW_THE_EVALUATION_WORKS.md)** — what we evaluate, the method, the two scoring layers, GPU/CUDA setup, and how to read results.
- **[USING_PROMPTFOO.md](USING_PROMPTFOO.md)** — hands-on guide to promptfoo (config, assertions, commands).

## Layout

```
deploy_local_models_v5.py   Deploys the 3 models on the GPU (CUDA build + -ngl offload)
rephrase-eval/              The AI Rephrase evaluation
  promptfooconfig.yaml      Wires up the 3 models, checks, and the Claude judge
  tests.yaml                Dataset: inputs + values each must preserve
  checks.js                 Deterministic checks (Layer 1)
  prompts/                  The production system prompt + prompt builder
  run_on_server.sh          Ship to server + run the eval
  pull_report.sh            Pull results + build/open the HTML report
  make_report.py            Builds the self-contained report.html
```

## Workflow

```bash
# One-time: deploy the models on the GPU server
scp deploy_local_models_v5.py ml-server:~/
ssh ml-server "python3 ~/deploy_local_models_v5.py --wait-minutes 5"

# Each eval run (two commands):
cd rephrase-eval
./run_on_server.sh      # runs the eval on the server
./pull_report.sh        # pulls results, builds report.html, archives a timestamped copy
```

## Models under test

| Model | Port | Notes |
|---|---|---|
| gemma-4-e2b | 8011 | |
| qwen3.5-2b | 8002 | |
| llama-3.2-3b-instruct | 8003 | leading candidate: clean output, fastest |

## Notes

Secrets (`.env.local`, e.g. the OpenRouter judge key) and generated output (`results.*`, `report.html`, `reports/`) are gitignored and never committed.
