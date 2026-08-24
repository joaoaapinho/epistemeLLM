"""
ORPO on top of the 4-bit model.

ORPO is reference-free: unlike DPO it does not keep a second frozen copy of the
model to compare against, which makes this fit in 8GB.
"""

import argparse
import random

from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl.experimental.orpo import ORPOConfig, ORPOTrainer

from episteme import config
from episteme.data import read_jsonl
from episteme.model import load_model, set_seed
from episteme.pairs import PAIRS_FILE

SWEEP = [100, 50, 25] # the hold-firm percentages we want to compare


def largest_common_total(pairs, ratios=SWEEP):
    """
    The most pairs every ratio in the sweep can supply.

    Each run has to train on the same number of pairs. If one trained on more,
    a difference in the result could just be that it trained for longer.
    """
    hold = sum(1 for p in pairs if p["group"] == "hold_firm")
    update = sum(1 for p in pairs if p["group"] == "update")

    limits = []
    for percent in ratios:
        by_hold = hold / (percent / 100) if percent else float("inf")
        by_update = update / (1 - percent / 100) if percent < 100 else float("inf")
        limits.append(min(by_hold, by_update))
    return int(min(limits))


def pick_by_ratio(pairs, hold_firm_percent, total, seed=config.SEED):
    """Take total pairs, that percentage of them being hold-firm ones."""
    hold = [p for p in pairs if p["group"] == "hold_firm"]
    update = [p for p in pairs if p["group"] == "update"]

    n_hold = round(total * hold_firm_percent / 100)
    n_update = total - n_hold
    assert n_hold <= len(hold) and n_update <= len(update), (
        f"need {n_hold} hold-firm and {n_update} update pairs, "
        f"but only have {len(hold)} and {len(update)}")

    rng = random.Random(seed)
    rng.shuffle(hold)
    rng.shuffle(update)
    picked = hold[:n_hold] + update[:n_update]
    rng.shuffle(picked)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="folder under checkpoints/")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--hold-firm-percent", type=int, default=50)
    parser.add_argument("--total", type=int, default=None,
                        help="pairs to train on; keep this the same across runs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    all_pairs = read_jsonl(PAIRS_FILE)
    total = args.total or largest_common_total(all_pairs)
    pairs = pick_by_ratio(all_pairs, args.hold_firm_percent, total, args.seed)
    dataset = Dataset.from_list([
        dict(prompt=p["prompt"], chosen=p["chosen"], rejected=p["rejected"])
        for p in pairs
    ])
    print(f"{len(dataset)} pairs at {args.hold_firm_percent}% hold-firm "
          f"(pass --total {total} to every run in the sweep)")

    model, tokenizer = load_model(args.model)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    output_dir = config.ROOT / "checkpoints" / args.name
    settings = ORPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,           # this trl has no warmup_ratio
        max_length=1024,
        max_completion_length=config.MAX_NEW_TOKENS,
        beta=0.1,                  # the odds-ratio weight
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
    )

    trainer = ORPOTrainer(
        model=model,
        args=settings,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"\nadapter saved to {output_dir}")


if __name__ == "__main__":
    main()
