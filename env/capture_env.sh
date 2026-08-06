#!/usr/bin/env bash
# Environment fingerprint. Output is committed so every result is traceable.
echo "# Environment"
echo
echo "Captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo '```'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -1
python - <<'PY'
import torch, importlib
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cc", torch.cuda.get_device_capability())
for m in ("vllm", "llmcompressor", "transformers", "lm_eval"):
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "NOT INSTALLED")
PY
echo '```'
echo
echo "sm_120 note: record any TORCH_CUDA_ARCH_LIST / kernel fallback warnings here verbatim."
