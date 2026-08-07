# Write-up outline: "Does 4-bit quantization break tool calling?"
Follow-up study to the NVFP4 post. Zero em dashes. Every number slot fills from
results/toolcall-<variant>.json; nothing is claimed before the JSON exists.

## Hypothesis
Structured output cracks before prose. The first study showed 4-bit damage
pooling in near-tie digit choices (numeric fidelity down 6 to 9 points while
suite means barely moved). Tool calling is the setting where a near-tie token
choice becomes a hard failure: one wrong digit in an account_id is not a
slightly worse sentence, it is the wrong API call. Prediction, stated before
the numbers exist: L1 and L2 hold roughly flat under quantization, and damage
concentrates in L4 argument values, worst in the extraction category and the
transfer_funds numeric-precision items, tracking each variant's numeric
fidelity delta from the first study.

## Method (one section, mostly pointers)
- 300 frozen items, four categories: selection (near-miss distractor tools),
  extraction (values embedded in phrasing: "$1,240.50", "half past nine"),
  abstention (no offered tool applies; correct behaviour is prose, not a
  forced call), compound (nested objects and arrays from one message).
- Gold generated from templates and seed-pinned lexicons, no LLM, disjoint
  from calibration data by construction. Frozen into the same manifest with
  the same refuse-to-refreeze behaviour as every other split.
- Hierarchical scoring, each layer conditional on the previous: parse,
  selection, schema validity, exact values (numeric tolerance 1e-9, digit
  strings string-exact on purpose). Bootstrap 95% CIs, 10k resamples.
- Five variants, same checkpoints as the first study, hermes tool parser,
  temperature 0, kernel receipt captured per run.

## Results skeleton (fill from JSONs)
- The five-by-four headline table. NUMBER SLOTS.
- Per-category breakdown for whichever variant degrades most. NUMBER SLOT.
- The transfer_funds slice specifically: L4 on amount and account_id versus
  each variant's numeric-fidelity score from study one. If they track, that is
  the connective finding: same damage, two surfaces. NUMBER SLOT.
- Abstention rates: does quantization make models trigger-happy (calling tools
  on general-knowledge questions)? This is the agent-safety angle. NUMBER SLOT.
- Native NVFP4 versus forced Marlin again: if the two kernel paths disagree on
  L4 while agreeing on L1-L3, the kernel-receipt argument from the first post
  extends to agent stacks unchanged.

## Honest-nulls section (pre-committed)
If the bet holds and every variant parses, selects, validates, and fills
arguments at BF16 rates within CIs, that is the publishable finding: 4-bit
quantization is safe for tool calling at 8B scale on this battery, and the
prose-first intuition about where damage lands survives contact with
structured output. Nulls get the same table and the same prominence. The
harness shipping with a null is still a harness nobody else has published.

## Scope confession
One model family, one size, synthetic battery of 12 tools, single-turn calls,
hermes parser only. Multi-turn tool chains, parallel calls, and larger tool
menus are follow-ups. Parser-version sensitivity is a known confound: the
parser is part of the measured system, which is the point of receipts.
