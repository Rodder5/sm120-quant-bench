# sm120-quant-bench

NVFP4 and W4A16 quantization on consumer Blackwell (RTX 5090, sm_120):
recipes, an eval harness designed to detect quantization damage that
aggregate benchmarks hide, and fully reproducible numbers.

**Claim under test:** 4-bit quantization damage is not uniform. It pools in
specific capabilities (numeric reasoning, code generation, long-context
retrieval) while suite averages and perplexity smooth it over. This repo
measures where it pools, per format, on hardware anyone can buy.

## Why sm_120

Consumer Blackwell is no longer unmeasured territory. A January 2026 study
([arXiv:2601.09527](https://arxiv.org/html/2601.09527v1)) benchmarks NVFP4, W4A16
and MXFP4 across the RTX 5060 Ti, 5070 Ti and 5090 with confidence intervals and a
released harness. This repo asks two questions that work leaves open.

**Does "NVFP4 on a 5090" measure what it claims to?** It depends on the stack, and
the only disclosure is a log line. On vLLM 0.24 with a ModelOpt mixed checkpoint,
NVFP4 layers fall back to the Marlin weight-only kernel with a stderr warning
("Your GPU does not have native support for FP4 computation")
([vllm#47749](https://github.com/vllm-project/vllm/issues/47749)) — 16-bit
activations, the card's FP4 hardware unused. On this repo's stack (vLLM 0.26,
compressed-tensors W4A4 checkpoint), the native path engages:
`Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`. Same GPU, same format
name, different machine measured. Published consumer-Blackwell NVFP4 numbers that
don't quote their kernel-selection logs may be measuring Marlin in a trench coat —
arXiv:2601.09527 does not document which path served its numbers, and from aggregate
reporting it is not answerable. Every eval and speed run here records vLLM's
kernel-selection lines into its results JSON (`kernel_selection`) and serve stderr
(`serve/logs_<variant>.stderr`); no number is reported without them.

**Can aggregate metrics see quantization damage at all?** Prior work reports
suite-level deltas ("2–4% quality loss"). The harness here reports per-capability
probes with bootstrap CIs and frozen splits carved before any quantization — and the
headline result (see table) is a case where perplexity calls two quantization
algorithms identical (19.66 vs 19.73) while a numeric-fidelity probe separates them
by 6.7 points.

Same card anyone can buy; the contribution is the measurement discipline, not the
silicon.

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
weight offload. It was never the pipeline — with `max_seq_length` omitted, calibration
pads every batch to its longest sample, and transformers' `masking_utils` materializes
the attention-mask broadcast over the whole padded batch: 128 samples × 3994² (our
longest sample, measured) at 8 bytes is 15.9 GiB — the "16.00 GiB" in the traceback.
Truncated to 2048 like the other recipes, that broadcast shrinks ~4× and calibration
runs beside 4 GB of co-tenants. Moral for 32 GB parts: when weight offload does not
move an OOM at all, the allocation is activations, not weights. Serving surfaced a
third, sm_120-specific quirk: vLLM auto-selects flashinfer for native FP4 compute on
consumer Blackwell (the alternatives in its NVFP4 kernel registry either run
weight-only via Marlin — 16-bit activations, not the format under test — or unoptimized
emulation), and that kernel (`fp4_gemm_cutlass_sm120`) is JIT-compiled at first load. A
conda nvcc without curand headers fails that build. Fix: put `curand_kernel.h` (shipped
in the `nvidia-*-cu13` pip wheels) on the include path, e.g. via the supported
`FLASHINFER_EXTRA_CUDAFLAGS` hook. Logs preserved.

**NVFP4 speed caveat:** NVFP4 TTFT/ITL numbers were measured with
`VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm` (default kernel tactics). Full fp4_gemm
autotune on sm_120 tunes every capture-size × layer-shape combination at ~1.5 min
each — 3.5 hours in, it was still going, so evals (where tactics change nothing)
and speed (where they matter at the margin) both skip it; a fully-autotuned speed
rerun is a cheap follow-up once the cache finishes building. Quality numbers are
unaffected by tactic selection.

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
