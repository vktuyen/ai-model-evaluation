#!/usr/bin/env bash
# =============================================================================
# Pull the latest eval results off the server, build the HTML report, and
# archive a timestamped copy.
#
# Run this AFTER ./run_on_server.sh has finished the eval on the server.
#
#   ./run_on_server.sh     # 1. runs the eval on the server
#   ./pull_report.sh       # 2. fetches results + builds/opens the report
#
# Produces (in this folder):
#   results.json, results.csv, report.html   <- latest
#   reports/<timestamp>/                      <- archived snapshot of each run
# =============================================================================
set -euo pipefail

SSH_TARGET="ml-server"
REMOTE_DIR="~/rephrase-eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Pulling results from ${SSH_TARGET}:${REMOTE_DIR}"
# tar-over-ssh (no rsync dependency)
if ! ssh "$SSH_TARGET" "cd ${REMOTE_DIR} && tar czf - results.json results.csv 2>/dev/null" \
      | tar xzf - -C "$SCRIPT_DIR" 2>/dev/null; then
  echo "ERROR: could not fetch results. Did ./run_on_server.sh finish successfully?"
  exit 1
fi

echo "==> Building report.html"
python3 "${SCRIPT_DIR}/make_report.py" "${SCRIPT_DIR}/results.json" "${SCRIPT_DIR}/report.html"

# Archive a timestamped snapshot so you keep a history of every run.
TS="$(date +%Y-%m-%d_%H%M%S)"
ARCHIVE="${SCRIPT_DIR}/reports/${TS}"
mkdir -p "$ARCHIVE"
cp "${SCRIPT_DIR}/results.json" "${SCRIPT_DIR}/results.csv" "${SCRIPT_DIR}/report.html" "$ARCHIVE/"

echo
echo "Done."
echo "  Latest report : ${SCRIPT_DIR}/report.html"
echo "  Archived copy : ${ARCHIVE}/report.html"
echo

# Open it automatically (macOS 'open', Linux 'xdg-open') — ignore if neither.
if command -v open >/dev/null 2>&1; then
  open "${SCRIPT_DIR}/report.html" || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${SCRIPT_DIR}/report.html" || true
fi
