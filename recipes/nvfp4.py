"""NVFP4 one-shot with llm-compressor.
The novel leg: consumer Blackwell (sm_120) NVFP4 behaviour is thinly documented.
Record EVERY kernel-fallback warning vLLM emits at load; that goes in the write-up.
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-calib", type=int, default=512)
    args = ap.parse_args()

    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft") \
        .shuffle(seed=args.seed).select(range(args.num_calib))

    # NVFP4: 4-bit float, hardware scale factors. Global scales need calibration data
    # (unlike plain FP8 dynamic). Keep lm_head and any draft head in BF16.
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
    oneshot(model=args.model, dataset=ds, recipe=recipe,
            num_calibration_samples=args.num_calib, output_dir=args.out)
    print(f"[nvfp4] saved -> {args.out}")

if __name__ == "__main__":
    main()
