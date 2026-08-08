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
