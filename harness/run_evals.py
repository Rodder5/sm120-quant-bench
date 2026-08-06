"""Eval orchestrator. One invocation = one model variant = one results JSON.
All tasks run against the frozen splits through one shared vLLM instance;
plain completion prompts, no chat template, identical across variants — the
comparison is the experiment, so formatting must not vary. Results JSON embeds
git hash + split manifest seed, so every published number is traceable or it
doesn't exist.
"""
import argparse, datetime, json, pathlib, random, re, subprocess, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from probes import longctx, numeric


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNCOMMITTED"


def load_split(splits_dir, name):
    p = pathlib.Path(splits_dir) / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


def bootstrap_ci(scores, n_resamples=10_000, seed=3407):
    """95% CI on the mean of item-level scores. Deltas whose CIs span the
    baseline are noise; report the interval, not just the point."""
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(n_resamples))
    return [round(means[int(0.025 * n_resamples)], 4), round(means[int(0.975 * n_resamples)], 4)]


_LAST_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

GSM8K_SHOTS = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Per minute Weng earns 12 / 60 = 0.2 dollars. For 50 minutes she earned 0.2 x 50 = 10 dollars. The answer is 10.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: Betty has 100 / 2 = 50 dollars. Her grandparents gave her 15 x 2 = 30 dollars. In total she has 50 + 15 + 30 = 95 dollars. She needs 100 - 95 = 5 dollars more. The answer is 5.

"""


def eval_gsm8k(generate, items):
    prompts = [GSM8K_SHOTS + f"Question: {it['question']}\nAnswer:" for it in items]
    outs = generate(prompts, max_tokens=400, stop=["Question:"])
    scores = []
    for o, it in zip(outs, items):
        nums = _LAST_NUM.findall(o)
        pred = nums[-1].replace(",", "").rstrip(".") if nums else None
        gold = it["answer"].rstrip(".")
        try:
            ok = pred is not None and float(pred) == float(gold)
        except ValueError:
            ok = pred == gold
        scores.append(1 if ok else 0)
    return scores


def eval_mmlu(generate, items):
    letters = ["A", "B", "C", "D"]
    prompts = []
    for it in items:
        ch = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(it["choices"]))
        prompts.append(f"Question: {it['question']}\n{ch}\nAnswer with the letter only.\nAnswer:")
    outs = generate(prompts, max_tokens=3)
    scores = []
    for o, it in zip(outs, items):
        m = re.search(r"[ABCD]", o)
        scores.append(1 if (m and letters.index(m.group(0)) == it["answer"]) else 0)
    return scores


HUMANEVAL_HARNESS = """
import signal
def _timeout(signum, frame): raise TimeoutError()
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(8)
{code}
{test}
check({entry_point})
print("PASS-SENTINEL")
"""


def eval_humaneval(generate, items):
    """pass@1, greedy. Each candidate runs in an isolated python subprocess with
    a wall-clock alarm; anything but the sentinel on stdout is a fail."""
    prompts = [it["prompt"] for it in items]
    outs = generate(prompts, max_tokens=512, stop=["\ndef ", "\nclass ", "\nif __name__", "\nprint("])
    scores = []
    for o, it in zip(outs, items):
        code = it["prompt"] + o
        script = HUMANEVAL_HARNESS.format(code=code, test=it["test"], entry_point=it["entry_point"])
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script); path = f.name
        try:
            r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=15)
            scores.append(1 if "PASS-SENTINEL" in r.stdout else 0)
        except Exception:
            scores.append(0)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)
    return scores


def eval_ppl(llm, items):
    """Mean NLL/token over the frozen wikitext slice via prompt_logprobs.
    Returns (ppl, per-item mean-NLL list) — CI is computed on item NLLs."""
    import math
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    outs = llm.generate([it["text"] for it in items], sp)
    item_nlls, total_nll, total_tok = [], 0.0, 0
    for out in outs:
        nll = ntok = 0
        for lp in out.prompt_logprobs or []:
            if lp is None:
                continue
            tok_lp = next(iter(lp.values())).logprob
            nll -= tok_lp; ntok += 1
        if ntok:
            item_nlls.append(nll / ntok); total_nll += nll; total_tok += ntok
    return math.exp(total_nll / total_tok), item_nlls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.55,
                    help="vLLM gpu_memory_utilization; low default so the box's other tenants survive")
    ap.add_argument("--max-len", type=int, default=17408)   # 16k probe + question + headroom
    ap.add_argument("--skip", default="",
                    help="comma-separated metrics to skip (e.g. longctx when VRAM is shared: "
                         "run it separately with the full window and merge)")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    manifest = json.loads((pathlib.Path(args.splits) / "MANIFEST.json").read_text())

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, revision=args.revision, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len,
              enforce_eager=False, seed=3407)

    def generate(prompts, max_tokens, stop=None):
        sp = SamplingParams(temperature=0, max_tokens=max_tokens, stop=stop)
        outs = llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    results = {"tag": args.tag, "model": args.model, "revision": args.revision,
               "git": git_hash(), "when": datetime.datetime.utcnow().isoformat() + "Z",
               "split_manifest_seed": manifest["seed"], "metrics": {}}

    def record(name, scores, extra=None):
        m = {"value": round(sum(scores) / len(scores), 4), "n": len(scores),
             "ci95": bootstrap_ci(scores)}
        if extra:
            m.update(extra)
        results["metrics"][name] = m
        print(f"[eval:{args.tag}] {name} = {m['value']} (n={m['n']}, ci {m['ci95']})")

    if "gsm8k" not in skip:
        record("gsm8k", eval_gsm8k(generate, load_split(args.splits, "gsm8k")))
    if "mmlu_stem" not in skip:
        record("mmlu_stem", eval_mmlu(generate, load_split(args.splits, "mmlu_stem")))
    if "mmlu_hum" not in skip:
        record("mmlu_hum", eval_mmlu(generate, load_split(args.splits, "mmlu_hum")))
    if "humaneval" not in skip:
        record("humaneval", eval_humaneval(generate, load_split(args.splits, "humaneval")))
    if "numeric" not in skip:
        record("numeric", numeric.run(generate, load_split(args.splits, "numeric")))
    if "longctx" not in skip:
        lc_scores, lc_depth = longctx.run(generate, load_split(args.splits, "longctx"))
        record("longctx", lc_scores, extra={"by_depth": lc_depth})
    if "ppl_wikitext" not in skip:
        ppl, item_nlls = eval_ppl(llm, load_split(args.splits, "ppl_wikitext"))
        results["metrics"]["ppl_wikitext"] = {"value": round(ppl, 4), "n": len(item_nlls),
                                              "nll_ci95": bootstrap_ci(item_nlls)}
        print(f"[eval:{args.tag}] ppl_wikitext = {ppl:.4f}")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{args.tag}.json"
    if p.exists():   # partial-run merge: existing metrics survive unless re-run
        prior = json.loads(p.read_text())
        prior_metrics = prior.get("metrics", {})
        prior_metrics.update(results["metrics"])
        results["metrics"] = prior_metrics
        results["merged_from"] = prior.get("git")
    p.write_text(json.dumps(results, indent=2))
    print(f"[eval:{args.tag}] wrote {p}")


if __name__ == "__main__":
    main()
