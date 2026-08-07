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


Baseline: `Qwen/Qwen3-8B` @ revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16.
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
| Perplexity (wikitext-2, frozen slice) | 17.924 | 19.662 | 19.725 | TODO |
| GSM8K (strict-match) | 91.8 [89.4, 94.2] | 88.4 [85.6, 91.2] | 88.8 [86.0, 91.4] | TODO |
| HumanEval (pass@1) | 63.4 [56.1, 70.7] | 64.6 [57.3, 72.0] | 58.5 [51.2, 65.8] | TODO |
| MMLU – STEM subset | 71.8 [68.6, 74.8] | 69.0 [65.8, 72.1] | 66.8 [63.5, 69.9] | TODO |
| MMLU – humanities subset | 62.3 [58.9, 65.6] | 62.6 [59.2, 65.9] | 60.2 [56.8, 63.6] | TODO |
| Long-context retrieval @16k | 100.0 | 100.0 | 100.0 | TODO |
| Long-context v2, multi-needle @16k | 96.7 | 93.3 | 95.0 | TODO |
| Numeric fidelity probe | 78.7 [74.0, 83.3] | 76.7 [71.7, 81.3] | 70.0 [65.0, 75.0] | TODO |
| Weights on disk (GB) | 16.4 | 6.1 | 6.1 | 6.4 |
| TTFT p50 (ms) | 14.8 | 7.7 | 7.7 | TODO |
| ITL p50 (ms/token) | 11.1 | 4.7 | 4.7 | TODO |

Deltas vs BF16 with bootstrap 95% CIs: `results/` after a full run.

**NVFP4 postmortem (2026-08-07):** the column exists because two independent bugs
were unpicked, and the debugging trail is worth more than the numbers. On the released
stack (llm-compressor 0.12.0.1, torch 2.11, one RTX 5090) the sequential pipeline
cannot fx-trace Qwen3-8B (its traced subgraph trips `You must specify exactly one of
input_ids or inputs_embeds`); that is fixed on llm-compressor main (unreleased), which
`recipes/nvfp4.py` now requires. Behind it hid a second failure that had been
misattributed to the quantizer across three earlier attempts: a recurring
"Tried to allocate 16.00 GiB" OOM that survived both `pipeline="basic"` and full CPU
weight offload. It was never the pipeline — with `max_seq_length` omitted, a single
10k+-token calibration conversation explodes the attention-mask expansion in
transformers' `masking_utils`. Truncated to 2048 like the other recipes, calibration
runs beside 4 GB of co-tenants. Moral for 32 GB parts: when weight offload does not
move an OOM at all, the allocation is activations, not weights. Serving surfaced a
third, sm_120-specific quirk: vLLM's NVFP4 matmul on consumer Blackwell is a
flashinfer JIT kernel (`fp4_gemm_cutlass_sm120`) compiled at first load, and a conda
nvcc without curand headers fails that build — unlike the sampler, it cannot be
disabled by env var, because it IS the GEMM. Fix: put `curand_kernel.h` (shipped in
the `nvidia-*-cu13` pip wheels) on nvcc's include path. Logs preserved.

**Long-context probe status:** BF16 and W4A16-GPTQ both score 100% at every depth,
so the current single-needle design is saturated and cannot discriminate — a ceiling,
not a victory. The probe needs hardening (multiple needles, near-miss distractor keys,
answer synthesis rather than retrieval) before this row means anything. Reported as-is
because a saturated metric silently presented as "no degradation" is exactly the
failure mode this repo exists to call out.

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
