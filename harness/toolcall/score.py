"""Hierarchical scoring for the tool-calling split.

Four layers, each conditional on the previous, reported separately, because
the analysis question is WHERE damage lands, not just how much:

  L1 parse:     the response contains a syntactically valid tool call
                (for abstention items: correctly contains none)
  L2 selection: the right tool was chosen (or correct abstention)
  L3 schema:    arguments validate against the tool's JSON schema
  L4 values:    normalized exact match on argument values

Denominators are conditional: an item enters a layer only if it passed every
earlier layer and the layer applies (abstention items stop after L2, since a
correct abstention has no arguments to validate). Rates come with item-level
bootstrap 95% CIs, 10,000 resamples.

Strictness notes, deliberate and documented:
- Extra arguments the model invents fail L4 (gold lists the complete arg set).
- Numeric tolerance 1e-9 applies to number/integer-typed fields only. Fields
  whose schema type is string stay string-exact after case-folding, so a digit
  string like account_id must survive verbatim. That asymmetry is the point of
  the transfer_funds probe.
- Primitive arrays compare as multisets (attendee order is not signal).
  Arrays of objects compare as multisets of canonicalized objects.
- Free-text fields (title, memo, notes) match by case-folded containment: the
  gold phrase must appear in the produced value. Exact match there would
  measure paraphrase tolerance ("Incident Postmortem Meeting" for gold
  "incident postmortem"), which is not the damage under study. Wrong-topic
  values still fail. Found by the bf16 smoke run before any results existed.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schemas import TOOLS_BY_NAME  # noqa: E402

LAYERS = ["L1_parse", "L2_selection", "L3_schema", "L4_values"]
NUM_TOL = 1e-9


# -- response unpacking -------------------------------------------------------

def extract_calls(response):
    """Pull tool calls out of a raw /v1/chat/completions response dict.
    Returns (calls, hard_error) where calls is a list of
    {"name": str, "arguments": dict} and hard_error marks unparseable JSON in
    an attempted call (which is an L1 failure, distinct from calling nothing).
    """
    calls, hard_error = [], False
    try:
        msg = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return [], True
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        name = fn.get("name")
        raw = fn.get("arguments")
        if isinstance(raw, dict):          # some servers pre-parse
            calls.append({"name": name, "arguments": raw})
            continue
        try:
            calls.append({"name": name, "arguments": json.loads(raw or "{}")})
        except (json.JSONDecodeError, TypeError):
            hard_error = True
    return calls, hard_error


# -- schema validation (self-contained, no jsonschema dependency) -------------

def _validate(value, schema):
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return False
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                return False
        for k, v in value.items():
            if k in props and not _validate(v, props[k]):
                return False
        return True
    if t == "array":
        return isinstance(value, list) and all(
            _validate(v, schema.get("items", {})) for v in value)
    if t == "string":
        if not isinstance(value, str):
            return False
        return "enum" not in schema or value in schema["enum"]
    if t == "integer":
        # accept 3.0 for 3: the damage we care about is in the digits
        return (isinstance(value, bool) is False
                and isinstance(value, (int, float)) and float(value).is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    return True   # untyped: permissive


def validate_schema(call):
    tool = TOOLS_BY_NAME.get(call["name"])
    if tool is None:
        return False
    return _validate(call["arguments"], tool["function"]["parameters"])


# -- value normalization ------------------------------------------------------

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})([ T](\d{2}):(\d{2}))?$")


def _norm(v):
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        s = v.strip()
        m = _DATE_RE.match(s.replace("T", " "))
        if m:
            return ("str", s.replace("T", " "))
        return ("str", s.casefold())
    if isinstance(v, list):
        return ("list", sorted((json.dumps(_norm(x), sort_keys=True) for x in v)))
    if isinstance(v, dict):
        return ("dict", {k: _norm(x) for k, x in sorted(v.items())})
    return ("other", v)


def _values_equal(got, want):
    ng, nw = _norm(got), _norm(want)
    if ng[0] == nw[0] == "num":
        return abs(ng[1] - nw[1]) <= max(NUM_TOL, NUM_TOL * abs(nw[1]))
    if ng[0] == nw[0] == "dict":
        if set(ng[1]) != set(nw[1]):
            return False
        return all(_values_equal(got[k], want[k]) for k in want)
    return ng == nw


FREE_TEXT_FIELDS = {"title", "memo", "notes"}


def _free_text_match(got, want):
    return (isinstance(got, str) and isinstance(want, str)
            and want.strip().casefold() in got.strip().casefold())


def values_match(call, expected_args):
    got = call["arguments"]
    if not isinstance(got, dict):
        return False
    if set(got) != set(expected_args):
        return False
    for k in expected_args:
        if k in FREE_TEXT_FIELDS:
            if not _free_text_match(got[k], expected_args[k]):
                return False
        elif not _values_equal(got[k], expected_args[k]):
            return False
    return True


# -- per-item scoring ---------------------------------------------------------

def score_item(gold, response):
    """Returns dict layer -> True/False/None (None = not reached or N/A)."""
    out = {l: None for l in LAYERS}
    calls, hard_error = extract_calls(response)
    abstain = gold["expected"]["tool"] is None

    if abstain:
        out["L1_parse"] = (not calls) and (not hard_error)
        out["L2_selection"] = out["L1_parse"] or None
        if out["L1_parse"]:
            out["L2_selection"] = True
        return out

    out["L1_parse"] = bool(calls) and not hard_error
    if not out["L1_parse"]:
        return out
    call = calls[0]                       # grade the first call made
    out["L2_selection"] = call["name"] == gold["expected"]["tool"]
    if not out["L2_selection"]:
        return out
    out["L3_schema"] = validate_schema(call)
    if not out["L3_schema"]:
        return out
    out["L4_values"] = values_match(call, gold["expected"]["args"])
    return out


# -- aggregation --------------------------------------------------------------

def bootstrap_ci(flags, n_boot=10000, seed=3407):
    """95% CI over item-level pass flags."""
    if not flags:
        return [None, None]
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choice(flags) for _ in range(n)) / n
                   for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 4),
            round(means[int(0.975 * n_boot)], 4)]


def aggregate(rows):
    """rows: list of (gold, layer_dict). Returns nested rates with CIs."""
    cats = sorted({g["category"] for g, _ in rows}) + ["overall"]
    table = {}
    for cat in cats:
        sub = [r for r in rows if cat == "overall" or r[0]["category"] == cat]
        table[cat] = {}
        for layer in LAYERS:
            flags = [1 if s[layer] else 0 for _, s in sub if s[layer] is not None]
            if not flags:
                table[cat][layer] = None
                continue
            table[cat][layer] = {
                "value": round(sum(flags) / len(flags), 4),
                "n": len(flags),
                "ci95": bootstrap_ci(flags),
            }
    return table


def render_text(table):
    lines = ["category      " + "".join(f"{l:>22}" for l in LAYERS)]
    for cat, layers in table.items():
        cells = []
        for l in LAYERS:
            m = layers.get(l)
            if m is None:
                cells.append(f"{'n/a':>22}")
            else:
                cells.append(f"{m['value']*100:5.1f} [{m['ci95'][0]*100:.1f},{m['ci95'][1]*100:.1f}] n={m['n']:<4}".rjust(22))
        lines.append(f"{cat:<14}" + "".join(cells))
    return "\n".join(lines)


def score_file(gold_path, raw_path):
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(gold_path)}
    rows = []
    for line in open(raw_path):
        rec = json.loads(line)
        g = gold[rec["id"]]
        rows.append((g, score_item(g, rec["response"])))
    return aggregate(rows), len(rows)


def _selftest():
    """Offline scorer sanity check against synthetic responses derived from
    gold. SELF-TEST ONLY: never a benchmark result, never written to results/.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_gold import generate
    items = generate(3407, 40)

    def resp_perfect(g):
        if g["expected"]["tool"] is None:
            return {"choices": [{"message": {"content": "Plain prose answer."}}]}
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": g["expected"]["tool"],
            "arguments": json.dumps(g["expected"]["args"])}}]}}]}

    def resp_broken(g, rng):
        mode = rng.randrange(3)
        if g["expected"]["tool"] is None or mode == 0:   # force a spurious/malformed call
            return {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "transfer_funds", "arguments": "{not json"}}]}}]}
        if mode == 1:                                    # wrong tool
            wrong = next(n for n in TOOLS_BY_NAME if n != g["expected"]["tool"])
            return {"choices": [{"message": {"tool_calls": [{"function": {
                "name": wrong, "arguments": "{}"}}]}}]}
        args = dict(g["expected"]["args"])               # corrupt one digit-ish value
        for k, v in args.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                args[k] = v + 1
                break
        else:
            args["invented_extra_arg"] = "x"
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": g["expected"]["tool"], "arguments": json.dumps(args)}}]}}]}

    rng = random.Random(7)
    perfect = aggregate([(g, score_item(g, resp_perfect(g))) for g in items])
    broken = aggregate([(g, score_item(g, resp_broken(g, rng))) for g in items])
    ok = all(m["value"] == 1.0 for lay in perfect.values() for m in lay.values() if m)
    print("[selftest] perfect responses -> all layers 1.0:", ok)
    print("[selftest] broken responses table (expect damage):")
    print(render_text(broken))
    if not ok:
        sys.exit("selftest FAILED: perfect responses did not score 1.0 everywhere")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold")
    ap.add_argument("--raw")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        table, n = score_file(args.gold, args.raw)
        print(render_text(table))
