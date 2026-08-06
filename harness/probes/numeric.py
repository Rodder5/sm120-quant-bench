"""Numeric fidelity probe. Multi-digit arithmetic, unit conversion, and
copy-the-number-faithfully tasks. Rationale: 4-bit damage concentrates in
tokens where near-tie logits decide between digits; prose masks it, this doesn't.
Scored strict string match on the final number."""
def run(model, splits):
    raise NotImplementedError("TODO: run frozen numeric items, strict-match scoring")
