#!/usr/bin/env bash
# Launch vLLM for a given variant. Capture stderr: kernel-fallback warnings on
# sm_120 are DATA for the write-up, not noise.
set -euo pipefail
V="$1"
case "$V" in
  bf16)        M="Qwen/Qwen3-8B"; EXTRA="--dtype bfloat16 --revision b968826d9c46dd6066d109eabc6255188de91218" ;;
  w4a16-gptq)  M="models/w4a16-gptq"; EXTRA="" ;;
  w4a16-awq)   M="models/w4a16-awq"; EXTRA="" ;;
  nvfp4)       M="models/nvfp4"; EXTRA="" ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac
exec vllm serve "$M" $EXTRA --max-model-len 16384 --port 8000 2> "serve/logs_$V.stderr"
