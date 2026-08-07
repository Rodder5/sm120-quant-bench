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

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft") \
        .shuffle(seed=args.seed).select(range(args.num_calib))
    # oneshot wants a text column; ultrachat ships chat 'messages' — render them
    # through the model's own chat template so calibration sees realistic inputs
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
                remove_columns=ds.column_names)

    # NVFP4: 4-bit float, hardware scale factors. Global scales need calibration data
    # (unlike plain FP8 dynamic). Keep lm_head and any draft head in BF16.
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
    # pipeline="basic": llm-compressor 0.12's sequential fx tracer cannot trace
    # Qwen3-8B (its wrapped input guard fires inside the traced subgraph). The
    # basic pipeline runs whole-model calibration instead — needs the card to
    # itself, which on a 32GB consumer part means evicting every co-tenant first.
    oneshot(model=args.model, dataset=ds, recipe=recipe,
            num_calibration_samples=args.num_calib, output_dir=args.out,
            pipeline="basic")
    print(f"[nvfp4] saved -> {args.out}")

if __name__ == "__main__":
    main()
