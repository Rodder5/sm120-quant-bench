"""Carve every eval split ONCE, before any quantization, and write a manifest.
Re-running after the manifest exists is refused: frozen means frozen.
This is the part of the repo that embodies the methodology argument.
"""
import argparse, hashlib, json, pathlib, sys, random

SPLITS = {
    # name: (source dataset, config, split, n_items)
    "ppl_wikitext":  ("wikitext", "wikitext-2-raw-v1", "test", 2000),
    "gsm8k":         ("gsm8k", "main", "test", 500),
    "humaneval":     ("openai_humaneval", None, "test", 164),   # full set, small anyway
    "mmlu_stem":     ("cais/mmlu", "all", "test", 800),          # filtered to STEM cats in loader
    "mmlu_hum":      ("cais/mmlu", "all", "test", 800),          # filtered to humanities cats
    "longctx":       ("synthetic", None, None, 200),             # generated needle/haystack, seed-pinned
    "numeric":       ("synthetic", None, None, 300),             # generated arithmetic/units probe
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest_p = out / "MANIFEST.json"
    if manifest_p.exists():
        sys.exit("MANIFEST.json exists: splits are frozen. Delete deliberately if you truly mean to re-carve.")

    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "splits": {}}
    for name, (src, cfg, split, n) in SPLITS.items():
        # TODO: materialize each split to JSONL here (real loaders / generators).
        # Every item gets a stable id; the manifest stores a content hash so any
        # later mutation of a split file is detectable.
        path = out / f"{name}.jsonl"
        path.touch()
        manifest["splits"][name] = {
            "source": src, "config": cfg, "split": split, "n": n,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_p.write_text(json.dumps(manifest, indent=2))
    print(f"[splits] frozen {len(SPLITS)} splits under {out} (seed {args.seed})")

if __name__ == "__main__":
    main()
