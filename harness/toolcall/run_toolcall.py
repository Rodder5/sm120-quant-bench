"""Orchestrator for the tool-calling eval.

Points at an already-running vLLM OpenAI-compatible server (launched with
--enable-auto-tool-choice --tool-call-parser hermes), replays the frozen
toolcall split against it, stores every raw response under results/raw/ so
scoring is re-runnable without re-serving, then scores and writes
results/toolcall-<tag>.json in the same envelope style as run_evals.py.

Greedy decoding: temperature 0, fixed seed, tool_choice auto (the abstention
category is meaningless if calls are forced).
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schemas import TOOLS_BY_NAME  # noqa: E402
from score import aggregate, render_text, score_item, LAYERS  # noqa: E402


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return None


def post_json(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_model_id(base_url):
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default=str(Path(__file__).parent.parent / "splits"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N items only (smoke testing)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    split_path = Path(args.splits) / "toolcall.jsonl"
    manifest_path = Path(args.splits) / "MANIFEST.json"
    if not split_path.exists():
        sys.exit(f"{split_path} missing: freeze the toolcall split first "
                 "(python harness/freeze_splits.py --add-toolcall ...)")
    manifest = json.loads(manifest_path.read_text())
    if "toolcall" not in manifest.get("splits", {}):
        sys.exit("toolcall split not in MANIFEST: refuse to run on unfrozen gold")

    gold = [json.loads(l) for l in open(split_path)]
    if args.limit:
        gold = gold[:args.limit]

    model = get_model_id(args.base_url)
    print(f"[toolcall:{args.tag}] serving model id: {model}; {len(gold)} items")

    raw_dir = Path(args.out) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"toolcall-{args.tag}.jsonl"

    rows, t0 = [], time.time()
    with open(raw_path, "w") as rawf:
        for i, g in enumerate(gold):
            tools = [TOOLS_BY_NAME[n] for n in g["tools_offered"]]
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": g["user_message"]}],
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
            }
            try:
                resp = post_json(f"{args.base_url}/v1/chat/completions", payload)
            except Exception as e:
                resp = {"error": f"{type(e).__name__}: {e}"}
            rawf.write(json.dumps({"id": g["id"], "response": resp}) + "\n")
            rows.append((g, score_item(g, resp)))
            if (i + 1) % 25 == 0:
                print(f"[toolcall:{args.tag}] {i + 1}/{len(gold)} "
                      f"({time.time() - t0:.0f}s)")

    table = aggregate(rows)
    print(render_text(table))

    # serve stderr is captured by serve/launch_vllm.sh; record where.
    variant = args.tag.replace("toolcall-", "")
    envelope = {
        "tag": f"toolcall-{args.tag}",
        "model_served": model,
        "git": git_hash(),
        "when": datetime.datetime.utcnow().isoformat() + "Z",
        "split_manifest_seed": manifest["seed"],
        "split_sha256": manifest["splits"]["toolcall"]["sha256"],
        "n_items": len(gold),
        "limited": bool(args.limit),
        "kernel_stderr": f"serve/logs_{variant}.stderr",
        "sampling": {"temperature": 0, "seed": args.seed,
                     "tool_choice": "auto", "max_tokens": args.max_tokens},
        "layers": LAYERS,
        "metrics": table,
    }
    out_path = Path(args.out) / f"toolcall-{args.tag}.json"
    out_path.write_text(json.dumps(envelope, indent=2))
    print(f"[toolcall:{args.tag}] wrote {out_path}"
          + (" (LIMITED smoke run, not a result)" if args.limit else ""))


if __name__ == "__main__":
    main()
