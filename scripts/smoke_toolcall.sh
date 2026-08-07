#!/usr/bin/env bash
# Smoke test for the tool-calling harness: full serving path against ONE
# variant (bf16), FIRST 10 items only, prints the four-layer table.
# Validates plumbing before burning GPU-hours. The limited-run results file is
# written under results/smoke/ and never mixes with real results.
#
# Usage: bash scripts/smoke_toolcall.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# offline scorer sanity first: costs nothing, catches scorer bugs early
python harness/toolcall/score.py --selftest

python harness/freeze_splits.py --seed 3407 --out harness/splits/ --add-toolcall 2>/dev/null \
  || echo "[smoke] toolcall split already frozen, good"

bash serve/launch_vllm.sh bf16 --tools &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do
  curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "[smoke] server died; see serve/logs_bf16.stderr"; exit 1; }
  sleep 10
done

python harness/toolcall/run_toolcall.py --tag bf16-smoke --out results/smoke/ --limit 10
echo "[smoke] done. If the table above shows sane L1-L4 numbers, run the real thing:"
echo "  ./reproduce.sh --stage toolcall-<variant>   for each of: bf16 w4a16-gptq w4a16-awq nvfp4 nvfp4-marlin"
