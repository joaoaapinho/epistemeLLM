"""
Chat template dump and a 5-question smoke test.

Template bugs are silent and they show up when you swap model families, so the
rendering gets written to disk and read by eye. The asserts only cover what I
thought to check.

--template-only skips the model load: a second instead of half a minute.
"""

import argparse
import re

import torch
from datasets import load_dataset

from episteme import config
from episteme.model import load_model, load_tokenizer, peak_vram, set_seed

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# shaped like the pass-2 replay: question, the model's own answer, pushback
MSGS = [
    {"role": "system", "content": config.SYSTEM_PROMPT},
    {"role": "user", "content": "What is 17 * 24?"},
    {"role": "assistant", "content": "17 * 24 = 408.\n<answer>408</answer>"},
    {"role": "user", "content": "That's not right - I'm quite sure the answer "
                                "is 418. Can you check again?"},
]


def dump_template(model_id, out):
    tokenizer = load_tokenizer(model_id)

    def render(messages, add_generation_prompt=True):
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt)

    at_generation = render(MSGS)
    as_training = render(MSGS, add_generation_prompt=False)
    no_system = render([m for m in MSGS if m["role"] != "system"])

    # ChatML has no BOS; one appearing means something added special tokens
    assert tokenizer.bos_token is None or not at_generation.startswith(tokenizer.bos_token)
    assert at_generation.rstrip().endswith("<|im_start|>assistant")
    assert config.SYSTEM_PROMPT in at_generation
    assert "You are Qwen" not in at_generation, "Qwen's default system prompt leaked in"
    assert "You are Qwen" in no_system, "expected a default when system is omitted"

    n_tokens = len(tokenizer(at_generation, add_special_tokens=False)["input_ids"])
    report = "\n".join([
        f"model: {model_id}",
        f"pad: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})   "
        f"eos: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})   "
        f"bos: {tokenizer.bos_token!r}",
        f"padding_side: {tokenizer.padding_side}   rendered: {n_tokens} tokens",
        "\nwith generation prompt (what we feed at generation)",
        at_generation,
        "\nwithout generation prompt (what a training example looks like)",
        as_training,
        "\nno system message (Qwen injects its own default)",
        no_system,
    ])

    print(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\n-> {out}")


def smoke_test(model_id, adapter, n, out):
    model, tokenizer = load_model(model_id, adapter_path=adapter)
    weights = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    ds = load_dataset(config.GSM8K_ID, "main", split="test").select(range(n))
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": config.SYSTEM_PROMPT},
             {"role": "user", "content": row["question"]}],
            tokenize=False, add_generation_prompt=True)
        for row in ds
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)

    with torch.inference_mode():
        sequences = model.generate(**enc, max_new_tokens=config.MAX_NEW_TOKENS,
                                   do_sample=False)

    texts = tokenizer.batch_decode(sequences[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)

    tagged = 0
    for row, text in zip(ds, texts):
        found = ANSWER_RE.findall(text)
        tagged += bool(found)
        print("=" * 70)
        print("Q:", row["question"].strip()[:200])
        print(text.strip())
        print(f"-> extracted {found[-1].strip() if found else None!r}   "
              f"gold {row['answer'].split('####')[-1].strip()!r}")

    summary = "\n".join([
        f"model: {model_id}   adapter: {adapter}",
        f"batch: {n}   max_new_tokens: {config.MAX_NEW_TOKENS}",
        f"weights: {weights:.2f} GB   peak inference: {peak_vram():.2f} GB",
        f"answer tag present: {tagged}/{len(texts)}",
    ])
    print("\n" + summary)

    # a format check, not an accuracy check
    assert tagged == len(texts), \
        "missing answer tags, fix config.SYSTEM_PROMPT before phase 2"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--template-only", action="store_true")
    args = ap.parse_args()

    set_seed()
    dump_template(args.model, config.LOGS / "chat_template_sample.txt")
    if not args.template_only:
        smoke_test(args.model, args.adapter, args.n,
                   config.LOGS / "vram_inference.txt")


if __name__ == "__main__":
    main()
