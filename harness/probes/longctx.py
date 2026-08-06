"""Long-context retrieval probe, 4k/8k/16k depths.
Needle-in-haystack with structured needles (key:value facts), answer scored
by exact match on the retrieved value. The point: KV-cache pressure and
quantization interact, and suite benchmarks rarely test past 4k. Seed-pinned
generation lives in freeze_splits; this file only *runs* the frozen items."""
import re

_NUM = re.compile(r"\d{8}")


def run(generate, items):
    """generate(prompts, max_tokens) -> list[str].
    Returns (per-item 0/1 list, per-depth accuracy dict)."""
    prompts = [it["context"] + "\n\n" + it["question"] + "\nAnswer: " for it in items]
    outs = generate(prompts, max_tokens=16)
    scores = []
    for o, it in zip(outs, items):
        m = _NUM.search(o)
        scores.append(1 if (m and m.group(0) == it["answer"]) else 0)
    by_depth = {}
    for s, it in zip(scores, items):
        by_depth.setdefault(it["depth"], []).append(s)
    return scores, {str(d): sum(v) / len(v) for d, v in sorted(by_depth.items())}
