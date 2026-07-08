#!/usr/bin/env bash
# =============================================================================
# Ship the FSM rephrase eval to the remote model server and run it there.
#
# Why on the server? The 3 models are deployed on the remote box, so promptfoo
# must run where it can reach http://10.30.11.110:8011|8002|8003. Running it on
# the server means those URLs resolve locally — no tunnel, no firewall holes.
#
# This script (run from your laptop):
#   1. copies this folder up to the server (rsync over your `ssh ml-server`)
#   2. ensures Node.js is installed on the server
#   3. runs `promptfoo eval` on the server (judge = OpenRouter / Claude)
#   4. copies the results (JSON + CSV) back into this local folder
#
# The OpenRouter key is read from .env.local (never shipped to the server;
# it is passed to the eval command only, over the encrypted SSH session).
#
# Usage:
#   ./run_on_server.sh
# =============================================================================
set -euo pipefail

# ---- Settings (uses your SSH host alias) ------------------------------------
SSH_TARGET="ml-server"            # your `ssh ml-server` alias
REMOTE_DIR="~/rephrase-eval"      # where the project lands on the server
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the OpenRouter judge key from the local-only env file.
if [ -f "${SCRIPT_DIR}/.env.local" ]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env.local"
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set. Add it to ${SCRIPT_DIR}/.env.local"
  exit 1
fi

echo "==> [1/4] Copying project to ${SSH_TARGET}:${REMOTE_DIR}"
# Use tar-over-ssh (no rsync dependency on either machine).
ssh "$SSH_TARGET" "mkdir -p ${REMOTE_DIR}"
tar czf - -C "$SCRIPT_DIR" \
  --exclude node_modules --exclude '.promptfoo' \
  --exclude '.env.local' --exclude 'results.json' --exclude 'results.csv' . \
  | ssh "$SSH_TARGET" "tar xzf - -C ${REMOTE_DIR}"

echo "==> [2/4] Ensuring Node.js on the server"
ssh "$SSH_TARGET" 'bash -lc "
  if ! command -v node >/dev/null 2>&1; then
    echo Installing Node.js 20...
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
  fi
  node --version
"'

echo "==> [3/3] Running promptfoo eval on the server"
# -tt forces a pseudo-terminal so promptfoo's live progress bar renders here.
ssh -tt "$SSH_TARGET" "bash -lc '
  cd ${REMOTE_DIR}
  for p in 8011 8002 8003; do
    curl -sf http://10.30.11.110:\$p/v1/models >/dev/null \
      && echo \"  endpoint \$p OK\" \
      || echo \"  WARNING: endpoint \$p not responding\"
  done
  export OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
  # -j 1 = one request at a time. On CPU-bound small models this is faster in
  # wall-clock than 4 concurrent (no core contention) AND gives clean, honest
  # per-model latency for the speed comparison.
  # --no-cache forces fresh generations (avoids stale/duplicated results from
  # earlier runs). promptfoo exits non-zero when any test fails; that is
  # expected during an eval, so do not let it abort before we pull results back.
  npx --yes promptfoo@latest eval -c promptfooconfig.yaml \
      -j 1 --no-cache --verbose -o results.json -o results.csv || true
'"

echo
echo "Eval finished on the server. Results are saved on ${SSH_TARGET}:${REMOTE_DIR}/"
echo
echo "Now pull them down and build the report with:"
echo "  ./pull_report.sh"
