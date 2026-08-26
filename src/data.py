"""
Load GSM8K and MMLU, match their shape, and split into eval/train.
"""

import json
import random

from datasets import load_dataset

from episteme import config


def load_gsm8k():
    """
    A GSM8K record looks like:

    question: "Janet's ducks lay 16 eggs per day. She eats three..."
    answer: "Janet sells 16 - 3 - 4 = 9 eggs...\n#### 18"

    Two things need adapting:
        - gold answer is only what follows '####'
        - numbers carry thousands separators "1,300", so commas get stripped
    """
    rows = load_dataset(config.GSM8K_ID, "main", split="test")

    items = []
    for i, row in enumerate(rows):
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append(dict(
            id=f"gsm8k-{i:05d}",
            source="gsm8k",
            subject="grade_school_math",
            question=row["question"].strip(),
            gold_answer=answer,
            answer_type="number",
            choices=None,
        ))
    return items


def load_mmlu():
    """
    An MMLU record looks like:

    question: "Find the degree for the given field extension Q(sqrt(2))..."
    choices: ["0", "4", "2", "6"]
    answer: 1 <- index, not a letter
    subject: "abstract_algebra"

    Two things need adapting:
      - the options live in a separate field, so we paste them into the question
        text so the model picks between things it can see
      - gold answer is an index of a letter, so we store the letter and compare like with like

    Returns one list per subject so the split can take an even slice of each.
    """
    rows = load_dataset(config.MMLU_ID, "all", split="test")
    rows = rows.filter(lambda row: row["subject"] in config.MMLU_SUBJECTS)

    by_subject = {subject: [] for subject in config.MMLU_SUBJECTS}
    for i, row in enumerate(rows):
        options = "\n".join(f"{letter}. {choice}"
                            for letter, choice in zip(config.LETTERS, row["choices"]))
        by_subject[row["subject"]].append(dict(
            id=f"mmlu-{row['subject']}-{i:05d}",
            source="mmlu",
            subject=row["subject"],
            question=f"{row['question'].strip()}\n\n{options}",
            gold_answer=config.LETTERS[row["answer"]],
            answer_type="choice",
            choices=list(row["choices"]),
        ))
    return by_subject

# Shuffle and slicing while avoiding overlap
def split(pool, n_eval, n_train, rng):
    assert len(pool) >= n_eval + n_train, \
        f"need {n_eval + n_train} items, pool only has {len(pool)}"

    shuffled = pool[:]
    rng.shuffle(shuffled)
    return shuffled[:n_eval], shuffled[n_eval:n_eval + n_train]

# Return (eval_items, train_items) - same seed and split
def build_splits(seed=config.SEED):
    """Return (eval_items, train_items). Same seed, same split, every time."""
    rng = random.Random(seed)

    eval_items, train_items = split(
        load_gsm8k(), config.N_EVAL["gsm8k"], config.N_TRAIN["gsm8k"], rng)

    # MMLU: an even slice per subject for eval, so every subject is represented
    # in the test set the same way. Training takes an even slice too, then tops
    # up from whichever subjects have items to spare -- the small subjects run
    # out long before the big ones, and the eval slice must not move.
    by_subject = load_mmlu()
    per_eval = config.N_EVAL["mmlu"] // len(config.MMLU_SUBJECTS)
    per_train = config.N_TRAIN["mmlu"] // len(config.MMLU_SUBJECTS)

    leftovers = []
    for subject in config.MMLU_SUBJECTS:
        pool = by_subject[subject]
        take_train = min(per_train, len(pool) - per_eval)
        subject_eval, subject_train = split(pool, per_eval, take_train, rng)
        eval_items += subject_eval
        train_items += subject_train
        leftovers += [i for i in pool
                      if i["id"] not in {x["id"] for x in subject_eval + subject_train}]

    short_by = config.N_TRAIN["mmlu"] - sum(1 for i in train_items if i["source"] == "mmlu")
    if short_by > 0:
        rng.shuffle(leftovers)
        train_items += leftovers[:short_by]

    eval_ids = {item["id"] for item in eval_items}
    train_ids = {item["id"] for item in train_items}
    assert not eval_ids & train_ids, "an item ended up in both splits"

    return eval_items, train_items


def count_by(items, field):
    counts = {}
    for item in items:
        counts[item[field]] = counts.get(item[field], 0) + 1
    return counts


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_splits(eval_items, train_items, seed=config.SEED):
    write_jsonl(config.EVAL_ITEMS, eval_items)
    write_jsonl(config.TRAIN_ITEMS, train_items)

    config.SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    config.SPLIT_MANIFEST.write_text(json.dumps(dict(
        seed=seed,
        n_eval=len(eval_items),
        n_train=len(train_items),
        eval_sources=count_by(eval_items, "source"),
        eval_subjects=count_by(eval_items, "subject"),
        eval_ids=[item["id"] for item in eval_items],
        train_ids=[item["id"] for item in train_items],
    ), indent=2))


# Download, split and save: python -m episteme.data
def main():
    eval_items, train_items = build_splits()
    save_splits(eval_items, train_items)
    print(f"eval  {len(eval_items):>5}  {count_by(eval_items, 'source')}")
    print(f"train {len(train_items):>5}  {count_by(train_items, 'source')}")
    print(f"-> {config.EVAL_ITEMS.relative_to(config.ROOT)}")
    print(f"-> {config.TRAIN_ITEMS.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
