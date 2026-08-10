# sm120-quant-bench

NVFP4 and W4A16 quantization on consumer Blackwell (RTX 5090, sm_120):
recipes, an eval harness designed to detect quantization damage that
aggregate benchmarks hide, and fully reproducible numbers. Write-up:
["NVFP4 on a 5090 names two different machines"](https://rodwoods.dev/posts/nvfp4-names-two-machines/).

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
| Perplexity (wikitext-2, frozen slice) | 17.924 | 19.662 | 19.725 | 18.282 |
| GSM8K (strict-match) | 91.8 [89.4, 94.2] | 88.4 [85.6, 91.2] | 88.8 [86.0, 91.4] | 86.6 [83.6, 89.6] |
| HumanEval (pass@1) | 63.4 [56.1, 70.7] | 64.6 [57.3, 72.0] | 58.5 [51.2, 65.8] | 65.2 [57.9, 72.6] |
| MMLU – STEM subset | 71.8 [68.6, 74.8] | 69.0 [65.8, 72.1] | 66.8 [63.5, 69.9] | 69.1 [65.9, 72.2] |
| MMLU – humanities subset | 62.3 [58.9, 65.6] | 62.6 [59.2, 65.9] | 60.2 [56.8, 63.6] | 59.6 [56.1, 63.0] |
| Long-context retrieval @16k | 100.0 | 100.0 | 100.0 | 100.0 |
| Long-context v2, multi-needle @16k | 96.7 | 93.3 | 95.0 | 93.3 |
| Numeric fidelity probe | 78.7 [74.0, 83.3] | 76.7 [71.7, 81.3] | 70.0 [65.0, 75.0] | 72.7 [67.7, 77.7] |
| Weights on disk (GB) | 16.4 | 6.1 | 6.1 | 6.4 |
| TTFT p50 (ms) | 14.8 | 7.7 | 7.7 | 9.3 |
| ITL p50 (ms/token) | 11.1 | 4.7 | 4.7 | 6.6 |

Deltas vs BF16 with bootstrap 95% CIs: `results/` after a full run.

**Download the checkpoints** (each ships its damage table and kernel receipt in the
model card): [W4A16-GPTQ](https://huggingface.co/Rodder5/Qwen3-8B-W4A16-GPTQ) ·
[W4A16-AWQ](https://huggingface.co/Rodder5/Qwen3-8B-W4A16-AWQ) ·
[NVFP4](https://huggingface.co/Rodder5/Qwen3-8B-NVFP4)

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

**NVFP4 speed caveat, resolved:** the table's NVFP4 TTFT/ITL were measured at
default kernel tactics (`VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm`), because full
fp4_gemm autotune on sm_120 costs hours (per capture-size × layer-shape, ~290
combinations measured). The promised fully-autotuned rerun has since landed
(`results/speed-nvfp4-tuned.json`): TTFT 9.3 ms, ITL 6.54 ms/token — statistically
identical to default tactics, and still behind the Marlin fallback's 7.3/4.6. Tactic
tuning does not close the native-path gap: at this model size, single-stream, the
deficit is structural. Quality numbers were never affected by tactics.

**Long-context probe status:** the v1 single-needle row sits at 100% for nearly
every variant — saturated, a ceiling rather than a victory, retained as the honesty
exhibit. The v2 row (three needles, near-miss distractor keys, queried needle never
last-inserted) de-saturates it and shows the degradation v1 could not: BF16 96.7 vs
GPTQ 93.3 at 16k depth. A saturated metric silently presented as "no degradation" is
exactly the failure mode this repo exists to call out.

**The one-table argument:** NVFP4 posts the best perplexity of the three quantized
formats (18.28, near baseline) and the worst GSM8K (86.6) with numeric fidelity 6
points under baseline. Ranked by the metric quantization papers lead with, you would
pick the format that damaged arithmetic most. Bonus finding: the same NVFP4
checkpoint forced through the Marlin weight-only fallback (`results/nvfp4-marlin.json`)
matches or slightly beats the native FP4 path on quality AND on single-stream speed
at default tactics — the "silent fallback" of vllm#47749 is, today, the stronger
serving path on this card. Caveats and kernel receipts in the results JSONs.

## Tool calling under quantization

Agent stacks bet that quantization preserves structured output: a model that
still writes fine prose is assumed to still emit valid, correctly-argued tool
calls. This study measures the bet, extending the numeric-fidelity finding
(4-bit damage pools in near-tie digit choices) to the place where a wrong
digit becomes a wrong API call. 300 frozen items across selection, extraction,
abstention, and compound categories; scoring is hierarchical so damage is
localized, not averaged: L1 did it emit a parseable call, L2 the right tool,
L3 schema-valid arguments, L4 exactly the right values. Same discipline as the
rest of the repo: frozen split in the manifest, gold generated from templates
and seed-pinned lexicons (no LLM), bootstrap 95% CIs, kernel receipts per run.

| Layer (conditional chain) | BF16 | W4A16 (GPTQ) | W4A16 (AWQ) | NVFP4 (native) | NVFP4 (Marlin) |
|---|---|---|---|---|---|
| L1 parse / correct abstain | 93.3 [90.3, 96.0] | 93.0 [90.0, 95.7] | 94.7 [92.0, 97.0] | 90.0 [86.7, 93.3] | 91.0 [87.7, 94.0] |
| L2 tool selection | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| L3 schema validity | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| L4 argument values | 91.7 [87.8, 95.1] | 89.7 [85.3, 93.6] | 88.0 [83.2, 92.3] | 93.3 [89.7, 96.4] | 90.9 [86.9, 94.4] |
| End-to-end success (all 300, unconditional) | 87.7 [84.0, 91.3] | 86.0 [82.0, 89.7] | 86.3 [82.3, 90.0] | 85.7 [81.7, 89.3] | 85.0 [80.7, 89.0] |

First read of the results (per-category tables in `results/toolcall-<variant>.json`,
raw responses in `results/raw/`, instrument caveats in NOTES.md):

- **The structural machinery never breaks.** Every variant scores 100 on tool
  selection and schema validity, near-miss distractors included, and 100 on
  abstention: no variant ever forces a spurious call on a question no tool
  answers. The agent-stack bet largely holds at 8B.
- **Digit copying survives 4-bit perfectly.** The transfer_funds slice
  (formatted amounts, 10-digit account ids) scores 100 for every variant. The
  pre-registered sharpest prediction was wrong, and instructively: study one's
  numeric damage lives in near-tie computation, and copying is not computation.
- **W4A16's L4 damage is mostly invented defaults, not garbled values.**
  Decomposing the failures: GPTQ and AWQ roughly double the baseline rate of
  hallucinated optional arguments (9.3% and 9.6% vs 5.4%), values the user
  never stated (repeat_daily: false, quality: 0.8), while their pure
  wrong-value counts sit at or below baseline. Both NVFP4 paths suppress
  invention to 1.5 to 2.5% instead. Format-specific damage with opposite
  signs. Genuine transformation fumbles exist too ("ten to noon" becomes
  10:00, "five to midnight" becomes 05:00) but they are the minority class.
- **NVFP4 does not stop calling; it concludes slower.** The max_tokens
  ablation (results/toolcall-ablation-maxtokens.json) re-ran every L1 failure
  at a doubled budget: 26 of NVFP4's 30 failures land, most correctly. The
  honest claim: FP4 weight quantization lengthens thinking-mode deliberation,
  so at a fixed 512-token budget NVFP4 fails to complete calls half again as
  often as baseline (30 vs 20), with a small never-concludes tail (4 vs 1).
  Both NVFP4 kernel paths show it, so it is the weights, not the activation
  path. Budgets are real in production; that is the finding.

Pairwise CIs overlap throughout; category-level gaps are directional, not
definitive, at n=300. The write-up treats them accordingly.

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
- `writeup/` – outline for the write-up this repo backs, published at
  [rodwoods.dev: "NVFP4 on a 5090 names two different machines"](https://rodwoods.dev/posts/nvfp4-names-two-machines/)

## Provenance rules

Splits are frozen before quantization and never touched again. Every results
JSON embeds the git hash, seed, and environment fingerprint it was produced
under. If a number in the write-up is not traceable to a JSON in `results/`,
it does not get published.

## License

MIT. <!-- TODO: confirm before pushing -->
