# sm120-quant-bench

NVFP4 and W4A16 quantization on consumer Blackwell (RTX 5090, sm_120):
recipes, an eval harness designed to detect quantization damage that
aggregate benchmarks hide, and fully reproducible numbers.

**Claim under test:** 4-bit quantization damage is not uniform. It pools in
specific capabilities (numeric reasoning, code generation, long-context
retrieval) while suite averages and perplexity smooth it over. This repo
measures where it pools, per format, on hardware anyone can buy.

## Why sm_120

Nearly all published NVFP4 work runs on datacenter parts (B200/B300, sm_100).
Consumer Blackwell is a different compute capability (sm_120) with its own
kernel coverage, toolchain quirks, and thinly documented behaviour. These
numbers were produced on one RTX 5090. Your mileage should not vary: that is
the point of `reproduce.sh`.

## Model

<!-- TODO: pin exact HF repo + revision hash -->
Baseline: `Qwen/Qwen3-8B` @ revision `TODO`, BF16.
Chosen because the BF16 baseline, both quantized variants, and the KV cache
for the long-context probe all fit a 32 GB card without offload, so every
comparison is apples-to-apples on identical hardware.

## Results

All numbers produced by `reproduce.sh` on the environment fingerprinted by
[`env/capture_env.sh`](env/capture_env.sh) (written to `env/environment.md` on
first run). Frozen splits carved once by
`harness/freeze_splits.py` (seed pinned) before any model was quantized.

| Metric | BF16 baseline | W4A16 (GPTQ) | W4A16 (AWQ) | NVFP4 |
|---|---|---|---|---|
| Perplexity (wikitext-2, frozen slice) | TODO | TODO | TODO | TODO |
| GSM8K (strict-match) | TODO | TODO | TODO | TODO |
| HumanEval (pass@1) | TODO | TODO | TODO | TODO |
| MMLU – STEM subset | TODO | TODO | TODO | TODO |
| MMLU – humanities subset | TODO | TODO | TODO | TODO |
| Long-context retrieval @16k | TODO | TODO | TODO | TODO |
| Numeric fidelity probe | TODO | TODO | TODO | TODO |
| Weights on disk (GB) | TODO | TODO | TODO | TODO |
| TTFT p50 (ms) | TODO | TODO | TODO | TODO |
| ITL p50 (ms/token) | TODO | TODO | TODO | TODO |

Deltas vs BF16 with bootstrap 95% CIs: `results/` after a full run.

**The one-table argument:** compare the spread of the MMLU rows against the
spread of the GSM8K and long-context rows. <!-- TODO: one sentence once numbers exist -->

## Reproduce

```bash
./reproduce.sh            # full pipeline: env capture -> splits -> baseline -> quantize x2 -> eval x3 -> speed
./reproduce.sh --stage eval-nvfp4   # or any single stage
```

Requires: RTX 5090 (or any sm_120 part), CUDA 12.8+, ~150 GB disk, one long day of GPU time.

## Layout

- `recipes/` – llm-compressor one-shot recipes, one file per target format
- `harness/` – split freezing, eval orchestration, and the three custom probes
- `serve/` – vLLM launch configs and the TTFT/ITL speed bench
- `env/` – environment capture (driver, CUDA, torch, vllm, llm-compressor versions)
- `results/` – raw JSON per run + rendered tables; nothing hand-edited
- `writeup/` – the blog post draft this repo backs

## Provenance rules

Splits are frozen before quantization and never touched again. Every results
JSON embeds the git hash, seed, and environment fingerprint it was produced
under. If a number in the write-up is not traceable to a JSON in `results/`,
it does not get published.

## License

MIT. <!-- TODO: confirm before pushing -->
