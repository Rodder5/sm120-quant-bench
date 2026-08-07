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

# Canonical MMLU category mapping (Hendrycks et al. categories.py, unchanged).
MMLU_STEM = {
    "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics",
    "computer_security", "conceptual_physics", "electrical_engineering",
    "elementary_mathematics", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
}
MMLU_HUM = {
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "professional_law", "world_religions",
}

# Filler vocabulary for the long-context haystack: common English words, fixed
# list so generation is reproducible across datasets-library versions.
_FILLER = ("the of and a to in is was he for it with as his on be at by had not "
           "are but from or have an they which one you were her all she there "
           "would their we him been has when who will more no if out so said "
           "what up its about into than them can only other new some could time "
           "these two may then do first any my now such like our over man me "
           "even most made after also did many before must through back years "
           "where much your way well down should because each just those people").split()


def _wikitext(ds, n, rng):
    lines = [t.strip() for t in ds["text"] if len(t.strip()) >= 100]
    rng.shuffle(lines)
    return [{"text": t} for t in lines[:n]]


def _gsm8k(ds, n, rng):
    idx = list(range(len(ds))); rng.shuffle(idx)
    return [{"question": ds[i]["question"],
             "answer": ds[i]["answer"].split("####")[-1].strip().replace(",", "")}
            for i in idx[:n]]


def _humaneval(ds, n, rng):
    return [{"task_id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for r in ds][:n]


def _mmlu(ds, n, rng, cats):
    rows = [r for r in ds if r["subject"] in cats]
    idx = list(range(len(rows))); rng.shuffle(idx)
    return [{"subject": rows[i]["subject"], "question": rows[i]["question"],
             "choices": rows[i]["choices"], "answer": int(rows[i]["answer"])}
            for i in idx[:n]]


def _longctx(n, rng):
    """Needle-in-haystack, structured needles, depths 4k/8k/16k tokens (approx by words)."""
    items = []
    depths = [4000, 8000, 16000]
    for i in range(n):
        depth = depths[i % 3]
        n_words = int(depth * 0.75)          # ~0.75 words/token for filler prose
        hay = [rng.choice(_FILLER) for _ in range(n_words)]
        key = f"needle-{rng.randrange(10**6):06d}"
        val = f"{rng.randrange(10**8):08d}"
        pos = rng.randrange(int(n_words * 0.05), int(n_words * 0.95))
        hay.insert(pos, f". The secret value for {key} is {val} .")
        items.append({"depth": depth, "context": " ".join(hay),
                      "question": f"What is the secret value for {key}? Answer with the number only.",
                      "answer": val})
    return items


def _numeric(n, rng):
    """Multi-digit arithmetic, unit conversion, faithful number copying.
    Strict string match on the final number: near-tie digit logits have nowhere to hide."""
    items = []
    for i in range(n):
        kind = i % 3
        if kind == 0:   # arithmetic
            a, b = rng.randrange(10**4, 10**7), rng.randrange(10**4, 10**7)
            op = rng.choice(["+", "-", "*"])
            ans = {"+": a + b, "-": a - b, "*": a * b}[op]
            q = f"Compute {a} {op} {b}. Answer with the number only."
        elif kind == 1:  # unit conversion, exact factors only
            v = rng.randrange(1, 10**5)
            unit = rng.choice([("km", "m", 1000), ("kg", "g", 1000),
                               ("hours", "minutes", 60), ("GB", "MB", 1000)])
            ans = v * unit[2]
            q = f"Convert {v} {unit[0]} to {unit[1]}. Answer with the number only."
        else:            # faithful copy
            ans = f"{rng.randrange(10**6, 10**9)}.{rng.randrange(10, 99)}"
            q = f"Repeat this number exactly, and nothing else: {ans}"
        items.append({"question": q, "answer": str(ans)})
    return items


def add_toolcall(out, manifest_p, seed):
    """Additive freeze for the tool-calling split. Existing entries and their
    hashes are preserved byte-for-byte; re-freezing toolcall is refused."""
    manifest = json.loads(manifest_p.read_text())
    if "toolcall" in manifest.get("splits", {}):
        sys.exit("toolcall split already frozen: frozen means frozen.")
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "toolcall"))
    from generate_gold import generate
    items = generate(seed, 300)
    path = out / "toolcall.jsonl"
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    manifest["splits"]["toolcall"] = {
        "source": "synthetic", "config": None, "split": None, "n": len(items),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_p.write_text(json.dumps(manifest, indent=2))
    print(f"[splits] toolcall: {len(items)} items frozen (seed {seed}); "
          "existing split entries untouched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-toolcall", action="store_true",
                    help="additively freeze ONLY the toolcall split into an "
                         "existing manifest; never touches other entries")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest_p = out / "MANIFEST.json"
    if args.add_toolcall:
        if not manifest_p.exists():
            sys.exit("--add-toolcall requires an existing MANIFEST.json")
        add_toolcall(out, manifest_p, args.seed)
        return
    if manifest_p.exists():
        sys.exit("MANIFEST.json exists: splits are frozen. Delete deliberately if you truly mean to re-carve.")

    from datasets import load_dataset

    manifest = {"seed": args.seed, "splits": {}}
    for name, (src, cfg, split, n) in SPLITS.items():
        rng = random.Random(f"{args.seed}:{name}")   # per-split stream: order-independent
        if src == "synthetic":
            items = _longctx(n, rng) if name == "longctx" else _numeric(n, rng)
        else:
            ds = load_dataset(src, cfg, split=split)
            items = {"ppl_wikitext": _wikitext, "gsm8k": _gsm8k, "humaneval": _humaneval}.get(
                name, lambda d, k, r: _mmlu(d, k, r, MMLU_STEM if name == "mmlu_stem" else MMLU_HUM)
            )(ds, n, rng)
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for i, item in enumerate(items):
                item["id"] = f"{name}-{i:05d}"
                f.write(json.dumps(item) + "\n")
        manifest["splits"][name] = {
            "source": src, "config": cfg, "split": split, "n": len(items),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"[splits] {name}: {len(items)} items")
    manifest_p.write_text(json.dumps(manifest, indent=2))
    print(f"[splits] frozen {len(SPLITS)} splits under {out} (seed {args.seed})")


if __name__ == "__main__":
    main()
