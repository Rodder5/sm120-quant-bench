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
