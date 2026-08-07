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
    ap.add_argument("--offload", action="store_true",
                    help="load the model with accelerate CPU offload so basic-pipeline "
                         "calibration fits a 32GB card (its whole-model pass otherwise "
                         "requests a second 16GiB on top of the weights)")
    ap.add_argument("--gpu-cap", default="12GiB")
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
    model = args.model
    if args.offload:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory={0: args.gpu_cap, "cpu": "96GiB"})
    oneshot(model=model, dataset=ds, recipe=recipe,
            num_calibration_samples=args.num_calib, output_dir=args.out,
            pipeline="basic")
    print(f"[nvfp4] saved -> {args.out}")

if __name__ == "__main__":
    main()
