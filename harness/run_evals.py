"""Eval orchestrator. One invocation = one model variant = one results JSON.
Standard tasks route through lm-eval; the two custom probes are local.
Results JSON embeds git hash + env fingerprint + split manifest hash,
so every published number is traceable or it doesn't exist.
"""
import argparse, json, pathlib, subprocess, datetime

def git_hash():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "UNCOMMITTED"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.loads((pathlib.Path(args.splits) / "MANIFEST.json").read_text())
    results = {"tag": args.tag, "model": args.model, "git": git_hash(),
               "when": datetime.datetime.utcnow().isoformat() + "Z",
               "split_manifest_seed": manifest["seed"], "metrics": {}}

    # --- standard tasks via lm-eval, vLLM backend ---
    # TODO: subprocess lm_eval with --model vllm, tasks gsm8k,humaneval,mmlu subsets,
    #       constrained to the frozen item ids (use --samples / task filter).
    # results["metrics"]["gsm8k"] = ...

    # --- custom probes ---
    from probes import longctx, numeric   # noqa: F401
    # TODO: results["metrics"]["longctx@16k"] = longctx.run(args.model, splits=args.splits)
    # TODO: results["metrics"]["numeric"]     = numeric.run(args.model, splits=args.splits)

    # --- bootstrap CIs ---
    # TODO: per-metric 95% CI via bootstrap over item-level correctness (n=10_000 resamples).
    #       Report delta-vs-baseline CI, not just point deltas: a delta whose CI spans 0 is noise.

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{args.tag}.json"
    p.write_text(json.dumps(results, indent=2))
    print(f"[eval:{args.tag}] wrote {p}")

if __name__ == "__main__":
    main()
