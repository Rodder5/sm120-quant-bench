# Write-up outline: "What does 'NVFP4 on a 5090' actually measure?"
Target: rodwoods.dev, ~2,000-2,600 words, the results table, two log excerpts as figures.
Post links repo in first paragraph. Zero em dashes. Every figure traces to a results JSON.

## 1. The hook: same card, same format name, two different machines measured (2 paragraphs)
- Open with the two log lines side by side. Line one, vLLM 0.24 with a ModelOpt mixed
  NVFP4 checkpoint on an RTX 5090: "Your GPU does not have native support for FP4
  computation", and the layer silently runs Marlin weight-only with 16-bit activations
  (vllm#47749). Line two, this repo's stack, vLLM 0.26 with a compressed-tensors W4A4
  checkpoint, same class of card: "Using FlashInferCutlassNvFp4LinearKernel for NVFP4
  GEMM". Native FP4 hardware, engaged.
- The question for the reader: both runs would be labelled "NVFP4 on a 5090" in a
  results table. How many published consumer-Blackwell NVFP4 numbers checked which
  of the two they were? One warning line in stderr is the only disclosure.

## 2. Prior art, engaged not ignored (2 paragraphs)
- arXiv:2601.09527 benchmarks NVFP4/W4A16/MXFP4 across the 5060 Ti, 5070 Ti and 5090
  with CIs, energy accounting and a released harness. Good work; cite it properly and
  summarize fairly (deployment economics, break-even framing).
- Two open edges, framed as questions and never as accusations: (a) the paper does not
  document which kernel served its NVFP4 numbers, and from aggregate reporting it is
  not answerable; (b) its quality claim is suite-level. This piece tests whether that
  framing can hide anything.

## 3. Benchmark truthfulness as an eval problem (3 paragraphs, the thesis)
- A benchmark is a claim about a system, and claims need provenance. Two ways a
  clean-looking table lies: it measured a different system than it named (kernel
  fallback, silent dtype coercion, wrong template), or it averaged away the damage
  (suite means, perplexity).
- Discipline used here: frozen splits carved before quantization (manifest + content
  hashes), per-capability probes, bootstrap CIs, and kernel-selection lines captured
  into every results JSON (`kernel_selection` field) plus serve stderr committed per
  variant. No number reported without its kernel receipt.
- The connective sentence: the same failure mode I hit in speech evals (a stale judge
  quietly inflating scores, see the judge-drift post) exists in quantization
  benchmarking as a stale label: the name in the table no longer matches the thing
  measured. Link the two posts.

## 4. What 4-bit actually costs, measured honestly (the table + 4 paragraphs)
- The results table verbatim (BF16 / W4A16-GPTQ / W4A16-AWQ / NVFP4, all columns
  populated, each traceable to a results JSON).
- Finding 1, the headline: PPL separates GPTQ and AWQ by 0.06 (19.662 vs 19.725,
  statistically nothing) while the numeric probe separates them by 6.7 points and
  drops AWQ 8.7 under baseline (78.7 -> 70.0). The aggregate says identical; the
  probes say AWQ damaged arithmetic. This is the one-table argument.
- Finding 2: damage pools. STEM and numeric degrade; humanities is flat (GPTQ even
  ties baseline, 62.6 vs 62.3; AWQ gives up 2.1). A 60-task mean would have diluted
  all of this below noise.
- Finding 3: the saturated-ruler catch. The v1 long-context probe scored 100 for
  every variant at every depth: a ceiling, not a victory. The v2 probe (three
  needles, near-miss distractor keys, queried needle never last-inserted)
  de-saturates it: BF16 96.7 / GPTQ 93.3 / AWQ 95.0 at 16k depth. Quantization does
  cost deep-context recall, and a saturated metric was hiding it. Keep the v1 row in
  the table, labelled, as the honesty exhibit.
- Finding 4: the napkin math held. 16.4 GB -> 6.1 GB is 2.69x fewer bytes per token;
  measured ITL went 11.1 -> 4.7 ms, 2.36x, i.e. 88% of the bandwidth-predicted
  ideal. Decode is a memory problem; here is the receipt. Honest nulls stay in:
  HumanEval deltas sit inside their CIs.

## 5. The experiment prior art cannot run: native vs the trench coat (2-3 paragraphs)
- Same NVFP4 checkpoint, same card, two serves: auto-selected native FlashInfer path
  vs forced Marlin weight-only (--linear-backend marlin). Report quality and speed
  deltas between what "NVFP4" names and what the fallback delivers.
  NUMBER SLOT: fill from results/nvfp4.json vs results/nvfp4-marlin.json.
- This turns section 1 from commentary into measurement: the cost of the silent
  fallback, quantified on the hardware where it silently happens.
- Also report the operational finding nobody documents: first-contact NVFP4 warmup
  on sm_120 is not minutes, it is roughly two hours of JIT compile plus per-shape
  autotune (~66 GEMM buckets at ~1.5 min each), one time, cached thereafter.

## 6. What this means for anyone quantizing for production (2 paragraphs)
- If your deployment target is consumer Blackwell, verify which kernel served you:
  check your stderr before your slides. The five one-line fixes it took to get the
  native path built on a stock toolchain (tracer, mask broadcast, curand headers,
  build parallelism, driver linker) are in the repo README postmortem.
- If your acceptance gate is perplexity or a suite mean, you cannot see the
  difference between two algorithms one of which broke arithmetic. Gate on the
  capabilities your users exercise.

## 7. Scope confession + next (1 paragraph)
- One model, one card, PTQ only, single-stream speed. Each is a follow-up.
  Invitation: reproduce.sh, file an issue if your numbers differ.

## Discipline notes (do not skip)
- Engage prior art generously; the criticism is of reporting norms, never of authors.
- Kernel question framed as an open question with evidence, not an accusation.
- Every number cites a results JSON tag; README table and post render from the same
  source. Zero em dashes. No KinSpeak content.
- Publish checklist: README section swapped (done) -> Marlin-forced run lands ->
  post live -> follow-up email to careers@cql.ca (2 short paragraphs: the table now
  has all four columns, the kernel-truthfulness finding, link, nothing else).
