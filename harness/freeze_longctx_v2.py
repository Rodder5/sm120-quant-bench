"""Second-generation long-context split. The v1 single-needle probe saturated
(100% for BF16 and both W4A16 variants at every depth), so it cannot detect
degradation. v2 raises the floor three ways:
  - THREE needles per item; the question asks for exactly one of them
  - near-miss distractor keys in the haystack (differ by one digit, different values)
  - the queried needle is never the last one inserted
The v1 split is untouched (frozen means frozen); this writes longctx_v2.jsonl
with its own manifest entry file.
"""
import argparse, hashlib, json, pathlib, random

_FILLER = ("the of and a to in is was he for it with as his on be at by had not "
           "are but from or have an they which one you were her all she there "
           "would their we him been has when who will more no if out so said "
           "what up its about into than them can only other new some could time "
           "these two may then do first any my now such like our over man me "
           "even most made after also did many before must through back years "
           "where much your way well down should because each just those people").split()


def gen(n, rng):
    items, depths = [], [4000, 8000, 16000]
    for i in range(n):
        depth = depths[i % 3]
        n_words = int(depth * 0.75)
        hay = [rng.choice(_FILLER) for _ in range(n_words)]
        needles = []
        for k in range(3):
            key = f"needle-{rng.randrange(10**6):06d}"
            val = f"{rng.randrange(10**8):08d}"
            needles.append((key, val))
            pos = rng.randrange(int(n_words * 0.05), int(n_words * 0.95))
            hay.insert(pos, f". The secret value for {key} is {val} .")
        # near-miss distractors: key differs by one digit, value differs entirely
        for key, _ in needles[:2]:
            digits = list(key.split('-')[1])
            j = rng.randrange(len(digits))
            digits[j] = str((int(digits[j]) + rng.randrange(1, 9)) % 10)
            near = f"needle-{''.join(digits)}"
            pos = rng.randrange(int(len(hay) * 0.05), int(len(hay) * 0.95))
            hay.insert(pos, f". The secret value for {near} is {rng.randrange(10**8):08d} .")
        qk, qv = needles[rng.randrange(2)]   # never the last-inserted needle
        items.append({"depth": depth, "context": " ".join(hay),
                      "question": f"What is the secret value for {qk}? Answer with the number only.",
                      "answer": qv})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=180)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    path = out / "longctx_v2.jsonl"
    man_p = out / "MANIFEST_longctx_v2.json"
    if man_p.exists():
        raise SystemExit("longctx_v2 already frozen; refusing to re-carve")
    rng = random.Random(f"{args.seed}:longctx_v2")
    items = gen(args.n, rng)
    with open(path, "w") as f:
        for i, it in enumerate(items):
            it["id"] = f"longctx_v2-{i:05d}"
            f.write(json.dumps(it) + "\n")
    man_p.write_text(json.dumps({"seed": args.seed, "n": len(items),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
    print(f"[longctx_v2] frozen {len(items)} items")


if __name__ == "__main__":
    main()
