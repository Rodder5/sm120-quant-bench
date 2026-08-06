# Write-up outline — "What 4-bit actually costs on a card you can buy"
Target: rodwoods.dev, ~1,800–2,400 words, one figure, one table. Post links repo in first paragraph.

## 1. The setup (2 paragraphs)
- Everyone ships 4-bit quantized models; nearly all published NVFP4 numbers come off
  datacenter Blackwell. I have the consumer part (5090, sm_120) and a question:
  what does 4-bit actually cost, and can the standard evals even see it?
- One model (Qwen3-8B @ pinned revision), three quantizations (W4A16-GPTQ, W4A16-AWQ,
  NVFP4), one frozen harness, everything reproducible from one script. [repo link]

## 2. Why the usual number is close to meaningless (the thesis, 3 paragraphs)
- Perplexity and suite averages are the README standard. Two failure modes:
  (a) averaging: damage pools in narrow capabilities and a 60-task mean dilutes it below noise;
  (b) contamination/leakage: unfrozen splits + calibration data drawn carelessly can overlap eval data.
- What a harness must do instead: frozen splits carved before any quantization
  (manifest + content hashes), calibration pool disjoint from eval by construction,
  per-capability reporting, deltas with bootstrap CIs so noise can't be narrated as signal.
- NUMBER SLOT: one concrete anecdote from the run where the aggregate moved <X but a
  capability moved >Y. This paragraph is the reason the post exists. If the run
  produces no such case, SAY SO — a null result honestly reported still demonstrates the method.

## 3. Recipes (2 paragraphs + code links)
- W4A16 x2 and NVFP4 via llm-compressor; what each format is in one sentence each;
  what stayed in BF16 (lm_head / draft heads) and why.
- sm_120 specifics: kernel coverage, every fallback warning verbatim, anything that
  differed from documented sm_100 behaviour. NUMBER SLOT: disk sizes, quantization wall-clock.

## 4. Results (the table + 4 paragraphs)
- The table from README, verbatim.
- Para per finding, ordered by size of effect. Candidate shapes (confirm against data, do not force):
  - math/code degrade before prose (or don't — report either way)
  - GPTQ vs AWQ disagree most on ___
  - NVFP4 vs W4A16 at same bit-width: quality gap vs speed gap
  - long-context: does damage grow with depth?
- NUMBER SLOTS throughout. Every claim cites a results JSON by tag.

## 5. Speed, honestly (2 paragraphs)
- TTFT/ITL per format, single-stream, same card. Quantization is a trade; report both sides.
- The "free lunch" check: where 4-bit was NOT faster (small batch, kernel fallback), say so.

## 6. What I'd do differently / what's next (1 paragraph)
- Scope confession: one model, one card, PTQ only, no MoE. Each is a follow-up post.
- Invitation: reproduce.sh, file an issue if your numbers differ.

## Discipline notes (do not skip)
- Zero em dashes. Canonical-numbers rule applies: every figure in the post traces to a
  results JSON; the post, README and any future CV line cite the same source.
- No KinSpeak content anywhere in this piece.
- Publish checklist: repo public -> post live -> send follow-up email to careers@cql.ca
  (2 short paragraphs, link, one specific finding, nothing else).
