"""Numeric fidelity probe. Multi-digit arithmetic, unit conversion, and
copy-the-number-faithfully tasks. Rationale: 4-bit damage concentrates in
tokens where near-tie logits decide between digits; prose masks it, this doesn't.
Scored strict string match on the final number."""
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text):
    hits = _NUM.findall(text)
    return hits[-1].replace(",", "") if hits else None


def run(generate, items):
    """generate(prompts, max_tokens) -> list[str]. Returns per-item 0/1 list."""
    outs = generate([it["question"] + "\nAnswer: " for it in items], max_tokens=48)
    return [1 if _last_number(o) == it["answer"] else 0 for o, it in zip(outs, items)]
