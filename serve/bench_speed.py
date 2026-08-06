"""TTFT / ITL bench against a locally served vLLM instance (serve/launch_vllm.sh).
Streams N fixed prompts, records time-to-first-token and inter-token latency,
reports p50s. Sequential requests on purpose: this measures the single-user
interactive regime, not saturated throughput.
"""
import argparse, json, pathlib, statistics, time, urllib.request

PROMPTS = [
    "Explain, step by step, how a bloom filter works and when its false positives matter.",
    "Write a Python function that merges two sorted lists without using sort().",
    "Summarize the causes of the 2008 financial crisis in one paragraph.",
    "Convert 4382 minutes into days, hours and minutes, showing your working.",
    "Describe the difference between TCP and UDP for someone who ships games.",
] * 4   # 20 requests


def stream_one(base, model, prompt, max_tokens):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, gaps, last = None, [], None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            elif last is not None:
                gaps.append(now - last)
            last = now
    return ttft, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    # resolve served model name
    with urllib.request.urlopen(f"{args.base}/v1/models", timeout=60) as r:
        model = json.load(r)["data"][0]["id"]

    # one warmup, unmeasured (first request pays compile/caching costs)
    stream_one(args.base, model, PROMPTS[0], 32)

    ttfts, all_gaps = [], []
    for p in PROMPTS:
        ttft, gaps = stream_one(args.base, model, p, args.max_tokens)
        if ttft is not None:
            ttfts.append(ttft)
        all_gaps.extend(gaps)

    out = {
        "tag": args.tag, "model": model, "n_requests": len(ttfts),
        "ttft_p50_ms": round(statistics.median(ttfts) * 1000, 1),
        "itl_p50_ms": round(statistics.median(all_gaps) * 1000, 2),
        "ttft_p95_ms": round(sorted(ttfts)[int(0.95 * len(ttfts)) - 1] * 1000, 1),
    }
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"speed-{args.tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[speed:{args.tag}] {out}")


if __name__ == "__main__":
    main()
