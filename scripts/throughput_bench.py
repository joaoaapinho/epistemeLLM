"""
Script to measure 4-bit generation speed. Find the largest batch size
that fits in VRAM and estimate how long a full eval will take.
"""

# Imports

import argparse
import gc
import json
import time
from pathlib import Path
 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

N_GENERATIONS_FULL_EVAL = 1600 

EVAL_MAX_NEW_TOKENS = 400
PROMPT_TOKENS_APPROX = 250


# Filler prompt
def build_prompts(tokenizer, batch_size, prompt_tokens):
    filler = "Solve this step by step. " * (prompt_tokens // 6)
    msgs = [{"role": "user", "content": filler}]
    text = tokenizer.apply_chat_template(msgs, tokenize = False, add_generation_prompt = True)
    return [text] * batch_size

# Gen call - return seconds and gen tokens
def run_once(model, tokenizer, batch_size, max_new_tokens, prompt_tokens):
    prompts = build_prompts(tokenizer, batch_size, prompt_tokens)
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens = max_new_tokens,
            min_new_tokens = max_new_tokens,
            do_sample=False,
            pad_token_id = tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    new_tokens = (out.shape[1] - enc["input_ids"].shape[1]) * batch_size
    return dt, new_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 24])
    ap.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    ap.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS_APPROX)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="logs/throughput.txt")
    args = ap.parse_args()
 
    assert torch.cuda.is_available(), "No CUDA device."
    dev = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: {dev}  ({total_vram:.1f} GB total)\n")
 
    print(f"Loading {args.model} in 4-bit NF4 ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Decoder-only models need LEFT padding for batched generation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model.eval()
 
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Weights resident: {weights_gb:.2f} GB\n")
 
    results = []
    for bs in args.batch_sizes:
        try:
            torch.cuda.reset_peak_memory_stats()
            # Warmup (not timed) - first call includes CUDA kernel compilation.
            run_once(model, tokenizer, bs, 32, args.prompt_tokens)
 
            times, toks = [], []
            for _ in range(args.reps):
                dt, n = run_once(
                    model, tokenizer, bs, args.max_new_tokens, args.prompt_tokens
                )
                times.append(dt)
                toks.append(n)
 
            peak = torch.cuda.max_memory_allocated() / 1e9
            med_t = sorted(times)[len(times) // 2]
            tok_s = toks[0] / med_t
            gen_s = bs / med_t   # completed generations per second
 
            results.append(
                dict(batch_size=bs, median_sec=med_t, tokens_per_sec=tok_s,
                     gens_per_sec=gen_s, peak_vram_gb=peak, ok=True)
            )
            print(f"  batch {bs:>3}  |  {med_t:6.2f} s  |  {tok_s:7.1f} tok/s  "
                  f"|  {gen_s:5.2f} gen/s  |  peak {peak:.2f} GB")
 
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:>3}  |  OOM - this is our ceiling")
            results.append(dict(batch_size=bs, ok=False))
            torch.cuda.empty_cache()
            gc.collect()
            break
 
    ok = [r for r in results if r.get("ok")]
    assert ok, "Every batch size OOMed. Drop to the 1B model or shorter prompts."
    best = max(ok, key=lambda r: r["gens_per_sec"])
 
    proj_sec = N_GENERATIONS_FULL_EVAL / best["gens_per_sec"]
    proj_min = proj_sec / 60
 
    print("\n" + "=" * 35)
    print(f"Best batch size: {best['batch_size']}")
    print(f"Throughput: {best['gens_per_sec']:.2f} generations/sec")
    print(f"Peak VRAM: {best['peak_vram_gb']:.2f} / {total_vram:.1f} GB")
    print(f"Full eval: {N_GENERATIONS_FULL_EVAL} generations")
    print(f"Projected duration: {proj_min:.1f} min")
    print("=" * 35)
 
    if proj_min > 60:
        print("\n> Gate Failed. Projection exceeds 60 min.")
        needed = int(N_GENERATIONS_FULL_EVAL * 60 / proj_min)
        print(f"At this speed, a 60-min budget allows ~{needed} generations.")
        print(f"Cut the eval set to roughly "
              f"{int(needed / (N_GENERATIONS_FULL_EVAL / 400))} items, or rent a GPU.")
    else:
        print("\n> Gate Passed.")
 
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            dict(device=dev, total_vram_gb=total_vram, model=args.model,
                 weights_gb=weights_gb, max_new_tokens=args.max_new_tokens,
                 sweep=results, best=best,
                 n_generations_full_eval=N_GENERATIONS_FULL_EVAL,
                 projected_minutes=proj_min,
                 gate_passed=bool(proj_min <= 60)),
            f, indent=2,
        )
    print(f"\nWritten to {args.out}")
 
 
if __name__ == "__main__":
    main()