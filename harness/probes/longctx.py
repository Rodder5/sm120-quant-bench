"""Long-context retrieval probe, 4k/8k/16k depths.
Needle-in-haystack with structured needles (key:value facts), answer scored
by exact key match. The point: KV-cache pressure + quantization interact,
and suite benchmarks never test past 4k. Seed-pinned generation lives in
freeze_splits; this file only *runs* the frozen items."""
def run(model, splits):
    raise NotImplementedError("TODO: serve via vLLM, query frozen items, score exact-match per depth")
