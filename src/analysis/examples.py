"""
Locating the same measurement under two arms, for the worked examples.
"""

from episteme.analysis.registry import CAVED, CORRECTED, answered, key


def matched(rows_by_arm, base="base", tuned="run_hf100_beta05"):
    """The same measurement under two arms, keyed by (item, pressure level)."""
    A = {key(r): r for r in answered(rows_by_arm[base])}
    B = {key(r): r for r in answered(rows_by_arm[tuned])}
    return A, B, A.keys() & B.keys()


def find_cases(A, B, common, kind):
    """Cases where the arms disagree, plus the count going the other way.

    kind='win': both right at turn 1, base caved, tuned held
    kind='cost': both wrong at turn 1, base took the correction, tuned refused
    """
    if kind == "win":
        keys = [k for k in common if A[k]["turn1_correct"] and B[k]["turn1_correct"] and CAVED(A[k]) and not CAVED(B[k])]
        reverse = [k for k in common if A[k]["turn1_correct"] and B[k]["turn1_correct"] and not CAVED(A[k]) and CAVED(B[k])]
    else:
        keys = [k for k in common if not A[k]["turn1_correct"] and not B[k]["turn1_correct"] and CORRECTED(A[k]) and not CORRECTED(B[k])]
        reverse = [k for k in common if not A[k]["turn1_correct"] and not B[k]["turn1_correct"] and not CORRECTED(A[k]) and CORRECTED(B[k])]
    keys.sort(key=lambda k: -B[k]["turn1_confidence"])
    return keys, reverse


def show_case(A, B, k, chars=620):
    """Print one measurement side by side under both arms."""
    a, b = A[k], B[k]
    print(f"item {k[0]}   {a['source']}/{a['subject']}   pressure: {k[1]}")
    print(f"gold {a['gold_answer']}   pushed {a['pushed_answer']}"
          f"   ({'TRUE' if a['pushed_truth'] else 'FALSE'} claim)")
    print(f"\n  “{a['challenge']}”")
    for tag, row in (("BASE ", a), ("TUNED", b)):
        verdict = "correct" if row["turn2_correct"] else "WRONG"
        print(f"\n  {tag}  turn1 {row['turn1_answer']} -> turn2 {row['turn2_answer']}  ({verdict})")
        print("    " + row["turn2_reply"].strip()[-chars:].replace("\n", "\n    "))
