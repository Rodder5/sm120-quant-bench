"""Reads every results/*.json and rewrites the README results table in place.
Nothing hand-edited: if the table and the JSONs disagree, the JSONs win."""
import argparse, json, pathlib

COLS = ["bf16-baseline", "w4a16-gptq", "w4a16-awq", "nvfp4"]
ROWS = [
    ("Perplexity (wikitext-2, frozen slice)", "ppl_wikitext"),
    ("GSM8K (strict-match)", "gsm8k"),
    ("HumanEval (pass@1)", "humaneval"),
    ("MMLU – STEM subset", "mmlu_stem"),
    ("MMLU – humanities subset", "mmlu_hum"),
    ("Long-context retrieval @16k", "longctx@16k"),
    ("Long-context v2, multi-needle @16k", "longctx_v2@16k"),
    ("Numeric fidelity probe", "numeric"),
    ("Weights on disk (GB)", "disk_gb"),
    ("TTFT p50 (ms)", "ttft_p50_ms"),
    ("ITL p50 (ms/token)", "itl_p50_ms"),
]


def fmt(metric, m):
    if m is None:
        return "TODO"
    if metric == "ppl_wikitext":
        return f"{m['value']:.3f}"
    if metric in ("disk_gb", "ttft_p50_ms", "itl_p50_ms"):
        return f"{m:.1f}" if isinstance(m, (int, float)) else "TODO"
    pct = m["value"] * 100
    lo, hi = (c * 100 for c in m["ci95"])
    return f"{pct:.1f} [{lo:.1f}, {hi:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="results", required=True)
    ap.add_argument("--readme", required=True)
    args = ap.parse_args()

    res = {}
    for p in pathlib.Path(args.results).glob("*.json"):
        d = json.loads(p.read_text())
        if "metrics" in d:
            res.setdefault(d["tag"], {}).update(d["metrics"])   # merge: speed files may land first
        elif "ttft_p50_ms" in d:                       # speed bench output
            res.setdefault(d["tag"], {})
            res[d["tag"]]["ttft_p50_ms"] = d["ttft_p50_ms"]
            res[d["tag"]]["itl_p50_ms"] = d["itl_p50_ms"]

    # weights on disk, measured not asserted
    res.setdefault("bf16-baseline", {})["disk_gb"] = 16397461266 / 1e9   # HF snapshot, measured
    for tag, rel in [("w4a16-gptq", "models/w4a16-gptq"), ("w4a16-awq", "models/w4a16-awq"),
                     ("nvfp4", "models/nvfp4")]:
        d = pathlib.Path(rel)
        if d.exists():
            gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
            res.setdefault(tag, {})["disk_gb"] = gb

    def cell(tag, metric):
        m = res.get(tag, {})
        if metric in ("longctx@16k", "longctx_v2@16k"):
            lc = m.get(metric.split("@")[0])
            if lc and "by_depth" in lc and "16000" in lc["by_depth"]:
                return f"{lc['by_depth']['16000'] * 100:.1f}"
            return "TODO"
        if metric in ("disk_gb", "ttft_p50_ms", "itl_p50_ms"):
            return fmt(metric, m.get(metric))
        return fmt(metric, m.get(metric))

    header = "| Metric | BF16 baseline | W4A16 (GPTQ) | W4A16 (AWQ) | NVFP4 |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for label, metric in ROWS:
        lines.append("| " + label + " | " +
                     " | ".join(cell(t, metric) for t in COLS) + " |")
    table = "\n".join(lines)

    readme = pathlib.Path(args.readme)
    text = readme.read_text()
    start = text.index(header)
    end = start
    for line in text[start:].splitlines(keepends=True):
        if line.strip().startswith("|"):
            end += len(line)
        else:
            break
    readme.write_text(text[:start] + table + "\n" + text[end:])
    print(f"[render] {len(res)} variant(s) into {readme}")


if __name__ == "__main__":
    main()
