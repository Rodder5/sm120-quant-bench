#!/usr/bin/env bash
# Full reproduction pipeline. Each stage is idempotent and re-runnable.
# Usage: ./reproduce.sh [--stage <name>]   stages: env, splits, baseline, quant-w4a16-gptq, quant-w4a16-awq, quant-nvfp4, eval-<variant>, speed, render
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-8B"
REV="b968826d9c46dd6066d109eabc6255188de91218"
SEED=3407
OUT="results"
STAGE="${2:-all}"
[ "${1:-}" = "--stage" ] || STAGE="all"

run() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

run env && {
  bash env/capture_env.sh > env/environment.md
  echo "[env] captured"
}

run splits && {
  python harness/freeze_splits.py --seed $SEED --out harness/splits/
  echo "[splits] frozen; do not re-run after this point"
}

run baseline && {
  python harness/run_evals.py --model "$MODEL" --revision "$REV" --dtype bfloat16 \
    --splits harness/splits/ --tag bf16-baseline --out $OUT/
}

run quant-w4a16-gptq && python recipes/w4a16_gptq.py --model "$MODEL" --revision "$REV" --seed $SEED --out models/w4a16-gptq
run quant-w4a16-awq  && python recipes/w4a16_awq.py  --model "$MODEL" --revision "$REV" --seed $SEED --out models/w4a16-awq
run quant-nvfp4      && python recipes/nvfp4.py      --model "$MODEL" --revision "$REV" --seed $SEED --out models/nvfp4

for v in w4a16-gptq w4a16-awq nvfp4; do
  run eval-$v && python harness/run_evals.py --model models/$v \
    --splits harness/splits/ --tag $v --out $OUT/
done

run speed && {
  for v in bf16 w4a16-gptq w4a16-awq nvfp4; do
    bash serve/launch_vllm.sh $v &
    SERVER=$!
    python serve/bench_speed.py --tag $v --out $OUT/
    kill $SERVER; wait $SERVER 2>/dev/null || true
  done
}

# -- tool calling under quantization ------------------------------------------
# Serve each variant with tool calling enabled (hermes parser), wait for
# health, replay the frozen toolcall split, score hierarchically.
# NVFP4 variants need the documented sm_120 env flags: the FP4 GEMM is a
# flashinfer JIT kernel and full autotune costs hours (README postmortem).
wait_healthy() {
  for _ in $(seq 1 90); do
    curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
    kill -0 "$1" 2>/dev/null || { echo "server died during startup"; return 1; }
    sleep 10
  done
  echo "server not healthy after 15 min"; return 1
}

run toolcall-splits && python harness/freeze_splits.py --seed $SEED --out harness/splits/ --add-toolcall

for v in bf16 w4a16-gptq w4a16-awq nvfp4 nvfp4-marlin; do
  run toolcall-$v && {
    case "$v" in nvfp4*)
      export VLLM_USE_FLASHINFER_SAMPLER=0
      export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm
      export MAX_JOBS=4
      export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}
    ;; esac
    bash serve/launch_vllm.sh $v --tools &
    SERVER=$!
    wait_healthy $SERVER
    python harness/toolcall/run_toolcall.py --tag $v --out $OUT/
    kill $SERVER; wait $SERVER 2>/dev/null || true
    sleep 10
  }
done

run render && python harness/render_table.py --in $OUT/ --readme README.md
echo "All requested stages complete. Numbers live in $OUT/, table rendered into README."
