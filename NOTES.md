# Toolchain and instrument notes

Running observations that are not results but shape how results should be read.

## Tool-calling study

### 2026-08-07: gold v2 refreeze, before any results existed

The bf16 smoke run (10 items, labelled not-a-result) caught two instrument
bugs. Both were fixed and the toolcall split was re-frozen ONCE
(sha256 cd6f3633... -> e33ebba6...), before any variant produced a scored
result, which is the only window where refreezing is legitimate. No other
split entry changed.

1. Gold ambiguity: the invoice template read "Bill {client} for $X". Qwen3
   parsed "Bill" as the client's first name ("Bill Lars Vestergaard"), which
   is a defensible reading of a sentence the generator wrote badly. Gold must
   not punish defensible parses. Template now reads "Invoice {client} for $X".
2. Scoring category error: exact match on free-text fields measured paraphrase
   tolerance, not quantization damage (model produced "Incident Postmortem
   Meeting" for gold title "incident postmortem"). Free-text fields (title,
   memo, notes) now match by case-folded containment; wrong-topic still fails.

### 2026-08-07: vLLM 0.26 hermes tool calling, first contact

- --enable-auto-tool-choice --tool-call-parser hermes worked as documented for
  Qwen3-8B on the first 10 items: tool_calls arrived structured, arguments as
  JSON strings, no parse failures, no thinking-mode leakage into tool_calls.
- vLLM's default gpu-memory-utilization (0.92) refuses to start on a card with
  ~3.7 GB of resident co-tenants. serve/launch_vllm.sh now accepts a GPU_MEM
  env override; the toolcall stages default it to 0.80.

### 2026-08-07: full-run observations (all five variants)

- Failure-class taxonomy from raw-response inspection, three distinct classes
  behind the L1/L4 numbers:
  1. Thinking-mode exhaustion (dominant L1 failure): the model emits <think>
     deliberation and exhausts the 512-token budget before any tool call.
     Worst on native NVFP4. CONFOUND: max_tokens=512 was an instrument choice;
     a robustness rerun at a higher budget would separate "deliberates longer
     under quantization" from "never concludes". Raw responses preserved for
     exactly this question.
  2. Transformation errors (dominant extraction L4 failure): spoken times
     mis-mapped ("ten to noon" to 10:00, "five to midnight" to 05:00), worst
     on AWQ, echoing its numeric-fidelity deficit.
  3. Invented optional arguments: values the user never stated (keep_aspect,
     quality, repeat_daily) supplied with hallucinated defaults. Scored as L4
     failures by design; agent stacks should care.
- Free-text containment still fails on dropped articles ("Renew car insurance"
  vs gold "renew the car insurance"). Affects all variants roughly equally, so
  comparisons stand, but absolute L4 rates understate value accuracy slightly.
  Recorded rather than patched: results exist now, and the instrument does not
  move after data collection.
- Abstention was 100 for every variant: no spurious calls, anywhere. The
  trigger-happiness worry did not materialize at 8B.

### 2026-08-07: the phantom tracer bug, closed

The night-one tracer failure on Qwen3-8B ("You must specify exactly one of
input_ids or inputs_embeds", llmcompressor 0.12.0.1) could not be reproduced
in any clean configuration: pinned 0.12.0.1 alone (CPU), pinned 0.12.0.1 +
compressed-tensors 0.17.0 (CPU), the same combo with the GPU visible, and the
same combo with max_seq_length omitted. The last of these fails, but with the
16 GiB mask OOM already filed as issue #3011, disguised on the release line by
append_autowrap_source_on_fail re-raising it under an autowrapped source dump.
Conclusion at the time: environmental, not filed. TRUE ENDING (2026-08-11,
found while answering a maintainer question on #3011): the tracer error never
existed. The night-one log's ValueError text was the autowrapped SOURCE DUMP
printed above the real exception, which was the same 16 GiB mask OOM as
everything else. One bug, presenting as three, each face created by error
presentation. The withdrawn draft was more right than the withdrawal notice:
there was nothing to file because there was nothing there. New facts fed back as a comment on #3011: the
allocation is invariant to sample count (8 vs 128 samples, identical
16.00 GiB), and the release version buries the OOM below generated code.

### 2026-08-07: toolcall scorer audit + receipts gap

- Audit of the perfect L1/L2/L3 scores (the rule: too-good numbers get audited
  first): near-miss sibling present in the offer list for 100% of engineered
  items; sampled selection passes are genuine choices against live
  distractors; abstention passes are real prose refusals; a systematic sweep
  of all abstention passes across all five variants found zero unparsed
  tool-call shapes hiding in content. The 100s stand.
- Receipts gap found and fixed: vllm serve emits kernel-selection INFO on
  stdout, and serve/launch_vllm.sh captured stderr only, so toolcall serve
  receipts landed in orchestrator logs instead of the repo. Receipts recovered
  into results/toolcall-<variant>.json (kernel_selection field); the serve
  script now captures both streams.
- Thinking mode: ON via the Qwen3 default chat template, identically
  configured for all five variants (same server flags). All tool calls in the
  study were emitted after a <think> block; the max_tokens=512 budget interacts
  with this, hence the ablation at 1024.
