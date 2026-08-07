"""NVFP4 one-shot with llm-compressor.
The novel leg: consumer Blackwell (sm_120) NVFP4 behaviour is thinly documented.
Record EVERY kernel-fallback warning vLLM emits at load; that goes in the write-up.

Two findings from getting this to run on a 32GB card, in order of discovery:
1. The released llm-compressor 0.12 sequential fx tracer cannot trace Qwen3-8B
   (its wrapped input guard fires inside the traced subgraph). Fixed on main —
   install `llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git`.
2. The recurring "Tried to allocate 16.00 GiB" OOM was never the quantizer: with
   max_seq_length omitted, one 10k+-token ultrachat conversation explodes the
   attention-mask expansion in transformers' masking_utils. Truncate like the
   other recipes do and the whole run fits beside 4GB of co-tenants.
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-calib", type=int, default=128)
    ap.add_argument("--max-seq", type=int, default=2048)
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
    oneshot(model=args.model, dataset=ds, recipe=recipe,
            max_seq_length=args.max_seq, num_calibration_samples=args.num_calib,
            output_dir=args.out)
    print(f"[nvfp4] saved -> {args.out}")

if __name__ == "__main__":
    main()
