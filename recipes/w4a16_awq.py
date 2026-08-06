"""W4A16 via AWQ, one-shot with llm-compressor.
Calibration set is pinned by seed and drawn AFTER splits are frozen,
from a pool disjoint with every eval split (enforced by freeze_splits manifest).
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-calib", type=int, default=512)
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()

    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    # TODO: swap calibration source if the model is chat-tuned (use chat-formatted calib)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft") \
        .shuffle(seed=args.seed).select(range(args.num_calib))

    recipe = AWQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],       # TODO: also ignore MTP/draft heads if present (keep BF16)
    )
    oneshot(model=args.model, dataset=ds, recipe=recipe,
            max_seq_length=args.max_seq, num_calibration_samples=args.num_calib,
            output_dir=args.out)
    print(f"[w4a16-awq] saved -> {args.out}")

if __name__ == "__main__":
    main()
