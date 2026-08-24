"""
Manual pairs read.

    python scripts/show_pairs.py first 3
    python scripts/show_pairs.py --n 10 first 10
    python scripts/show_pairs.py --group update
"""

import argparse
import json

from episteme import config
from episteme.pairs import PAIRS_FILE


def show(pair, width=600):
    question, first_answer, pushback = (pair["prompt"][1]["content"], pair["prompt"][2]["content"], pair["prompt"][3]["content"])
    print("=" * 78)
    print(f"{pair['id']}   [{pair['group']}]")
    print("-" * 78)
    print("QUESTION\n ", question[:width].replace("\n", "\n  "))
    print("\nIT ANSWERED\n ", first_answer[-width:].replace("\n", "\n  "))
    print("\nWE PUSHED BACK\n ", pushback)
    print("\nCHOSEN  (train toward this)\n ",
          pair["chosen"][0]["content"][-width:].replace("\n", "\n  "))
    print("\nREJECTED  (train away from this)\n ",
          pair["rejected"][0]["content"][-width:].replace("\n", "\n  "))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--group", choices=["hold_firm", "update"], default=None)
    parser.add_argument("--file", default=str(PAIRS_FILE))
    args = parser.parse_args()

    pairs = [json.loads(line) for line in open(args.file)]
    if args.group:
        pairs = [p for p in pairs if p["group"] == args.group]

    counts = {}
    for pair in pairs:
        counts[pair["group"]] = counts.get(pair["group"], 0) + 1
    print(f"{len(pairs)} pairs  {counts}\n")

    for pair in pairs[:args.n]:
        show(pair)


if __name__ == "__main__":
    main()
