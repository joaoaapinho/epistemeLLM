"""
Drop pairs whose chosen response just agrees with the user instead of
re-deriving the answer

E.g., "... Profit = $320 - $32 = $288. However, if the problem intends for the
profit to be $160 as suggested, then ... the correct profit based on the
given options is $160."   <answer>$160</answer>

The 50/50 arm learned nothing from data like this, since agreeing vs re-deriving isn't 
visible to the model without gold, so this keeps only pairs where chosen shows its
own working.

python scripts/filter_pairs.py
python scripts/filter_pairs.py --show 6
"""

import argparse
import re

from episteme import config
from episteme.data import read_jsonl, write_jsonl
from episteme.pairs import PAIRS_FILE

OUT_FILE = config.DATA / "train" / "pairs_derived.jsonl"

# Phrases that make the user's assertion the reason for the answer not reasoning.
DEFERENCE = re.compile(
    r"given the (options|provided|context)|as suggested|provided solution"
    r"|reference solution|the intended answer|based on the given"
    r"|if the problem (intends|means)|you(?:'re| are) (right|correct)"
    r"|as you (said|mentioned|pointed)|since you|you indicated"
    r"|according to (you|your)|the correct answer is indeed",
    re.I)

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)

# A choice response has to actually argue for something. Below this it is
# almost always a bare restatement of the letter.
MIN_CHOICE_WORDS = 40


def reply_text(value):
    """Pairs store either a string or a chat list; we want the last message."""
    return value if isinstance(value, str) else value[-1]["content"]


def working(text):
    """Everything before the answer tag, which is the part that has to support it."""
    tag = ANSWER_TAG.search(text)
    return text[:tag.start()] if tag else text


def states_answer(body, gold):
    """Does the gold value appear in the working, not only in the tag?"""
    gold = str(gold).strip()
    if gold in body:
        return True
    # 51.50 and 51.5 are the same answer written two ways
    if "." in gold:
        trimmed = gold.rstrip("0").rstrip(".")
        return len(trimmed) > 1 and trimmed in body
    return False


def verdict(pair, item):
    """Why this pair was kept or dropped, so the counts can be explained."""
    body = working(reply_text(pair["chosen"]))

    if DEFERENCE.search(body):
        return "deferred to the user"

    if item["answer_type"] == "number":
        if not states_answer(body, item["gold_answer"]):
            return "answer never appears in the working"
    elif len(body.split()) < MIN_CHOICE_WORDS:
        return "too short to be a derivation"

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=PAIRS_FILE)
    parser.add_argument("--out", default=OUT_FILE)
    parser.add_argument("--show", type=int, default=0,
                        help="print this many kept and dropped examples")
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    items = {item["id"]: item for item in read_jsonl(config.TRAIN_ITEMS)}

    kept, dropped = [], []
    for pair in pairs:
        reason = verdict(pair, items[pair["id"]])
        (dropped if reason else kept).append((pair, reason))

    groups = sorted({p["group"] for p in pairs})
    print(f"{'group':<12}{'kept':>7}{'of':>7}{'share':>8}")
    for group in groups:
        before = [p for p in pairs if p["group"] == group]
        after = [p for p, _ in kept if p["group"] == group]
        print(f"{group:<12}{len(after):>7}{len(before):>7}{len(after)/len(before):>8.0%}")

    smallest = min(len([p for p, _ in kept if p["group"] == g]) for g in groups)
    print(f"\nlargest balanced run: {smallest * len(groups)} pairs {smallest} per group)")
    print("pass that number as --total so every arm trains on the same amount")

    print("\nwhy pairs were dropped")
    reasons = {}
    for _, reason in dropped:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"{count:>5} {reason}")

    write_jsonl(args.out, [p for p, _ in kept])
    print(f"\n-> {args.out}")

    for label, rows in (("KEPT", kept), ("DROPPED", dropped)):
        for pair, reason in rows[:args.show]:
            body = reply_text(pair["chosen"]).strip()
            print(f"\n{'=' * 78}\n{label}  {pair['id']}  group={pair['group']}"
                  f"{'' if reason is None else ' reason: ' + reason}")
            print(f"gold: {items[pair['id']]['gold_answer']}")
            print("chosen:", body[-460:].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
