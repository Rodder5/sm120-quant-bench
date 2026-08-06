"""Numeric fidelity probe. Multi-digit arithmetic, unit conversion, and
copy-the-number-faithfully tasks. Rationale: 4-bit damage concentrates in
tokens where near-tie logits decide between digits; prose masks it, this doesn't.
Scored strict string match on the final number."""
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _first_number(text):
    """The FIRST number is the answer; models that answer correctly and then
    free-associate more arithmetic afterwards must not be scored on the ramble.
    (Verified against raw outputs: last-number scoring under-reported the BF16
    baseline by ~60 points.)"""
    m = _NUM.search(text)
    return m.group(0).replace(",", "") if m else None


def run(generate, items):
    """generate(prompts, max_tokens) -> list[str]. Returns per-item 0/1 list."""
    outs = generate([it["question"] + "\nAnswer: " for it in items], max_tokens=48)
    return [1 if _first_number(o) == it["answer"] else 0 for o, it in zip(outs, items)]
