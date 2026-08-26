"""
Build the preference pairs that ORPO trains on.

For every training item: ask the question, push back, and sample a few replies.
Some end on the right answer, some don't, making the signal:
    
- it was RIGHT and we pushed a lie: ending right means it held firm
- it was WRONG and we pushed the truth: ending right means it took the correction

Either way the reply that ends on the correct answer is the preferred one, so
one rule covers both situations.

Items where every sample went the same way are dropped - with no disagreement
there is no pair to make.
"""

import argparse
import random

from episteme import config
from episteme.data import read_jsonl, write_jsonl
from episteme.extract import extract_answer, is_correct
from episteme.generate import chat_prompt, generate
from episteme.model import load_model, set_seed
from episteme.templates import (PRESSURE_LEVELS, answer_to_push,
                                build_challenge, build_distractors)

PAIRS_FILE = config.DATA / "train" / "pairs.jsonl"
SAMPLES_FILE = config.DATA / "train" / "samples.jsonl"


def ask_once(model, tokenizer, items, batch_size):
    """First turn: just the question. Returns (reply_text, was_correct) per item."""
    prompts = [
        chat_prompt(tokenizer, [{"role": "system", "content": config.SYSTEM_PROMPT},
                                {"role": "user", "content": item["question"]}])
        for item in items
    ]
    replies = generate(model, tokenizer, prompts, batch_size=batch_size,
                       desc="first answers")

    results = []
    for item, reply in zip(items, replies):
        answer = extract_answer(reply, item["answer_type"], item["choices"])
        results.append((reply, is_correct(answer, item["gold_answer"],
                                          item["answer_type"], item["choices"])))
    return results


def sample_replies(model, tokenizer, prompts, n_samples, batch_size):
    """Several replies per prompt, so the samples can disagree with each other."""
    repeated = [prompt for prompt in prompts for _ in range(n_samples)]
    replies = generate(model, tokenizer, repeated, batch_size=batch_size,
                       sample=True, desc="sampling replies")
    return [replies[i * n_samples:(i + 1) * n_samples] for i in range(len(prompts))]


def split_by_outcome(item, replies):
    """Sort replies into those ending on the right answer and those that don't."""
    good, bad = [], []
    for reply in replies:
        answer = extract_answer(reply, item["answer_type"], item["choices"])
        if answer is None:
            continue  # no answer tag, so we cannot tell which it is
        if is_correct(answer, item["gold_answer"], item["answer_type"], item["choices"]):
            good.append(reply)
        else:
            bad.append(reply)
    return good, bad


def build_pairs(samples, items_by_id, per_item=2):
    """
    Turn saved replies into chosen/rejected pairs.

    An item with 4 sampled replies often has more than one good and more than
    one bad, so it can supply more than one pair. per_item caps how many we take
    from a single question, so one item cannot dominate the training set.
    """
    pairs, dropped = [], {"no answer tag": 0, "every sample agreed": 0}

    for sample in samples:
        item = items_by_id[sample["id"]]
        good, bad = split_by_outcome(item, sample["replies"])

        if not good and not bad:
            dropped["no answer tag"] += 1
            continue
        if not good or not bad:
            dropped["every sample agreed"] += 1
            continue

        conversation = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["first_answer"]},
            {"role": "user", "content": sample["pushback"]},
        ]
        for i in range(min(per_item, len(good), len(bad))):
            pairs.append(dict(
                id=sample["id"], group=sample["group"], prompt=conversation,
                chosen=[{"role": "assistant", "content": good[i]}],
                rejected=[{"role": "assistant", "content": bad[i]}],
            ))
    return pairs, dropped


def report(pairs, dropped, n_items):
    counts = {}
    for pair in pairs:
        counts[pair["group"]] = counts.get(pair["group"], 0) + 1
    print(f"\n{len(pairs)} pairs from {n_items} items  {counts}")
    for reason, n in dropped.items():
        print(f"  dropped, {reason}: {n}")


def rebuild(per_item):
    """Redo the pairing from saved replies. No model, no GPU, a second or two."""
    samples = read_jsonl(SAMPLES_FILE)
    items_by_id = {i["id"]: i for i in read_jsonl(config.TRAIN_ITEMS)}
    pairs, dropped = build_pairs(samples, items_by_id, per_item)
    write_jsonl(PAIRS_FILE, pairs)
    report(pairs, dropped, len(samples))
    print(f"-> {PAIRS_FILE.relative_to(config.ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pairs-per-item", type=int, default=2,
                        help="an item with several good and bad replies can "
                             "supply more than one pair")
    parser.add_argument("--rebuild", action="store_true",
                        help="redo the pairing from data/train/samples.jsonl, "
                             "no model needed")
    parser.add_argument("--shuffle", action="store_true",
                        help="mix the sources before --limit; the file is "
                             "ordered gsm8k first, so a small limit is all gsm8k")
    args = parser.parse_args()

    if args.rebuild:
        rebuild(args.pairs_per_item)
        return

    set_seed()
    items = read_jsonl(config.TRAIN_ITEMS)
    if args.shuffle:
        random.Random(config.SEED).shuffle(items)
    items = items[:args.limit]
    distractors = build_distractors(items)
    model, tokenizer = load_model(args.model)

    first_answers = ask_once(model, tokenizer, items, args.batch_size)

    # Pick a pressure level per item at random, so the training data is not all
    # phrased the same way.
    rng = random.Random(config.SEED)
    conversations = []
    for item, (reply, was_correct) in zip(items, first_answers):
        pushed = answer_to_push(item, distractors[item["id"]], was_correct)
        conversations.append([
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": build_challenge(
                rng.choice(list(PRESSURE_LEVELS)), pushed)},
        ])

    prompts = [chat_prompt(tokenizer, c) for c in conversations]
    sampled = sample_replies(model, tokenizer, prompts, args.samples, args.batch_size)

    pairs, samples = [], []
    dropped = {"no answer tag": 0, "every sample agreed": 0}
    for item, (_, was_correct), conversation, replies in zip(
            items, first_answers, conversations, sampled):
        good, bad = split_by_outcome(item, replies)
        samples.append(dict(
            id=item["id"], group="hold_firm" if was_correct else "update",
            question=item["question"], gold_answer=item["gold_answer"],
            first_answer=conversation[2]["content"],
            first_was_correct=was_correct,
            pushback=conversation[3]["content"],
            replies=replies,
            n_ended_right=len(good), n_ended_wrong=len(bad),
            kept=bool(good and bad),
        ))
        if not good and not bad:
            dropped["no answer tag"] += 1
            continue
        if not good or not bad:
            # all the samples went the same way, so there is nothing to contrast
            dropped["every sample agreed"] += 1
            continue
        pairs.append(dict(
            id=item["id"],
            # what the model needed to do here, so we can vary the mix later
            group="hold_firm" if was_correct else "update",
            prompt=conversation,
            chosen=[{"role": "assistant", "content": good[0]}],
            rejected=[{"role": "assistant", "content": bad[0]}],
        ))

    write_jsonl(PAIRS_FILE, pairs)
    write_jsonl(SAMPLES_FILE, samples)
    counts = {}
    for pair in pairs:
        counts[pair["group"]] = counts.get(pair["group"], 0) + 1
    kept = len(pairs) / len(items) if items else 0
    print(f"\n{len(pairs)} pairs from {len(items)} items ({kept:.0%})  {counts}")
    for reason, n in dropped.items():
        print(f"  dropped, {reason}: {n}")
    print(f"-> {PAIRS_FILE.relative_to(config.ROOT)}")
    print(f"-> {SAMPLES_FILE.relative_to(config.ROOT)}  (every reply, kept or not)")


if __name__ == "__main__":
    main()
