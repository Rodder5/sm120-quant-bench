#!/usr/bin/env bash
# Launch vLLM for a given variant. Capture stderr: kernel-fallback warnings on
# sm_120 are DATA for the write-up, not noise.
set -euo pipefail
V="$1"
case "$V" in
  bf16)         M="Qwen/Qwen3-8B"; EXTRA="--dtype bfloat16 --revision b968826d9c46dd6066d109eabc6255188de91218" ;;
  w4a16-gptq)   M="models/w4a16-gptq"; EXTRA="" ;;
  w4a16-awq)    M="models/w4a16-awq"; EXTRA="" ;;
  nvfp4)        M="models/nvfp4"; EXTRA="" ;;
  nvfp4-marlin) M="models/nvfp4"; EXTRA="--linear-backend marlin" ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac
# Optional second arg --tools enables OpenAI-style tool calling (Qwen3 needs
# the hermes parser). Plain serving behaviour is unchanged without it.
if [ "${2:-}" = "--tools" ]; then
  EXTRA="$EXTRA --enable-auto-tool-choice --tool-call-parser hermes"
fi
# Optional GPU_MEM env caps gpu-memory-utilization: vLLM's 0.92 default needs
# a cleaner card than a box with resident co-tenants can offer.
if [ -n "${GPU_MEM:-}" ]; then
  EXTRA="$EXTRA --gpu-memory-utilization $GPU_MEM"
fi
# Both streams into the capture file: vllm serve emits kernel-selection INFO
# on stdout, tracebacks on stderr, and the receipt needs them all.
exec vllm serve "$M" $EXTRA --max-model-len 16384 --port 8000 > "serve/logs_$V.stderr" 2>&1
