"""
Measure how the model behaves under pushback.

turn 1: ask the question, see whether it got it right
turn 2: show it its own answer, contradict it, see whether it changes

Same shape as pairs.py, but greedy instead of sampled, and both pressure levels
instead of one.

To be run on base model and tuned adapter.
"""

import argparse
import json
import time

import torch

from episteme import config
from episteme.data import read_jsonl, write_jsonl
from episteme.extract import extract_answer, is_correct
from episteme.generate import chat_prompt, generate
from episteme.metrics import breakdown, bucket_summary, compute_metrics
from episteme.model import load_model, set_seed
from episteme.templates import (PRESSURE_LEVELS, answer_to_push,
                                build_challenge, build_distractors)

# Turn 1 - Greedy for the model to give the same answer
def ask(model, tokenizer, items, system_prompt, batch_size):
    prompts = [
        chat_prompt(tokenizer, [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": item["question"]}])
        for item in items
    ]
    replies = generate(model, tokenizer, prompts, batch_size=batch_size,
                       desc="turn 1")

    answers = []
    for item, reply in zip(items, replies):
        answer = extract_answer(reply, item["answer_type"], item["choices"])
        answers.append(dict(
            reply=reply,
            answer=answer,
            correct=is_correct(answer, item["gold_answer"],
                               item["answer_type"], item["choices"]),
        ))
    return answers

# Turn 2 (1x per pressure level) - Replay model's words back to it
def push_back(model, tokenizer, items, first, distractors, system_prompt, batch_size):
    rows, prompts = [], []

    for item, answer in zip(items, first):
        for level in PRESSURE_LEVELS:
            pushed = answer_to_push(item, distractors[item["id"]], answer["correct"])
            challenge = build_challenge(level, pushed)

            prompts.append(chat_prompt(tokenizer, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": answer["reply"]},
                {"role": "user", "content": challenge},
            ]))
            rows.append(dict(
                id=item["id"], source=item["source"], subject=item["subject"],
                pressure_level=level,
                # tell the truth only when it was wrong
                pushed_truth=not answer["correct"],
                pushed_answer=str(pushed),
                gold_answer=item["gold_answer"],
                turn1_answer=None if answer["answer"] is None else str(answer["answer"]),
                turn1_correct=answer["correct"],
                turn1_reply=answer["reply"],
                challenge=challenge,
            ))

    replies = generate(model, tokenizer, prompts, batch_size=batch_size, desc="turn 2")

    item_by_id = {item["id"]: item for item in items}
    for row, reply in zip(rows, replies):
        item = item_by_id[row["id"]]
        answer = extract_answer(reply, item["answer_type"], item["choices"])

        row["turn2_answer"] = None if answer is None else str(answer)
        row["turn2_reply"] = reply
        row["turn2_correct"] = is_correct(answer, item["gold_answer"],
                                          item["answer_type"], item["choices"])
        row["changed_mind"] = row["turn2_answer"] != row["turn1_answer"]
        row["answered_both_turns"] = (row["turn1_answer"] is not None
                                      and row["turn2_answer"] is not None)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="folder under results/")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", default="base", choices=list(config.PROMPTS),
                        help="which system prompt to test")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    seed = set_seed()
    system_prompt = config.PROMPTS[args.prompt]
    items = read_jsonl(config.EVAL_ITEMS)[:args.limit]
    distractors = build_distractors(items)

    model, tokenizer = load_model(args.model, adapter_path=args.adapter)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    first = ask(model, tokenizer, items, system_prompt, args.batch_size)
    rows = push_back(model, tokenizer, items, first, distractors,
                     system_prompt, args.batch_size)

    minutes = (time.perf_counter() - started) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    results = compute_metrics(rows)
    results["by_source"] = breakdown(rows, "source")
    results["by_pressure_level"] = breakdown(rows, "pressure_level")

    outdir = config.RESULTS / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outdir / "responses.jsonl", rows)
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2))
    (outdir / "buckets.json").write_text(json.dumps(bucket_summary(rows), indent=2))
    (outdir / "run_info.json").write_text(json.dumps(dict(
        name=args.name, model=args.model, adapter=args.adapter, seed=seed,
        prompt=args.prompt, system_prompt=system_prompt,
        items=len(items), rows=len(rows), minutes=round(minutes, 1),
        peak_vram_gb=round(peak_gb, 2),
    ), indent=2))

    print(f"\ngot it right: {results['n_right']}   got it wrong: {results['n_wrong']}")
    for name in ("caved", "corrected", "accuracy_after", "changed_mind", "dug_in"):
        m = results[name]
        value = "n/a" if m["value"] is None else f"{m['value']:.3f}"
        print(f"  {name:<16} {value}   ({m['k']}/{m['n']})")
    gap = results["pressure_gap"]
    print(f"  flips when lied to  : {gap['flips_when_lied_to']}")
    print(f"  flips when told true: {gap['flips_when_told_truth']}")
    print(f"\n{minutes:.1f} min, peak {peak_gb:.2f} GB -> {outdir}")


if __name__ == "__main__":
    main()
