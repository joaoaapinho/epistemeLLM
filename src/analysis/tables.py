"""
The headline rates, the data-integrity checks, and the paired comparisons.
"""

import numpy as np
import pandas as pd
from scipy import stats

from episteme.analysis.registry import (ARMS, CAVED, CHANGED, CORRECTED, PRESSURE, answered, buckets, key, rate)


def headline(rows_by_arm):
    """One row per arm: the metrics every other section refers back to."""
    out = []
    for name, rows in rows_by_arm.items():
        a, right, wrong = buckets(rows)
        lie = [r for r in a if not r["pushed_truth"]]
        truth = [r for r in a if r["pushed_truth"]]
        caved, corrected = rate(right, CAVED), rate(wrong, CORRECTED)
        out.append(dict(
            arm=ARMS[name]["label"], family=ARMS[name]["family"],
            n_answered=len(a), turn1_accuracy=len(right) / len(a),
            caved=caved["value"], caved_lo=caved["lo"], caved_hi=caved["hi"],
            corrected=corrected["value"],
            corrected_lo=corrected["lo"], corrected_hi=corrected["hi"],
            accuracy_after=rate(a, CORRECTED)["value"],
            dug_in=rate(wrong, lambda r: not CHANGED(r))["value"],
            pressure_gap=rate(lie, CHANGED)["value"] - rate(truth, CHANGED)["value"],
        ))
    return pd.DataFrame(out).set_index("arm")


def reference_policies(head):
    """Two degenerate policies to read accuracy_after against.

    Always complying scores the turn-1 error rate, since the protocol pushes gold
    whenever the model was wrong. Never changing scores the turn-1 accuracy.
    """
    ref = head.copy()
    ref["if_never_changed"] = ref.turn1_accuracy
    ref["if_always_complied"] = 1 - ref.turn1_accuracy
    ref["cost_of_pushback"] = ref.accuracy_after - ref.if_never_changed
    return ref[["accuracy_after", "if_never_changed", "if_always_complied",
                "cost_of_pushback", "family"]]


def provenance(rows_by_arm, metrics, info):
    """Row counts, what the recovery pass did, and how much is missing."""
    out = []
    for name, rows in rows_by_arm.items():
        out.append(dict(
            arm=ARMS[name]["label"],
            rows=len(rows), expected=info[name].get("rows"),
            dropped_by_recovery=metrics[name].get("rows_dropped_after_recovery"),
            answered=sum(r["answered_both_turns"] for r in rows),
            elicited=sum("turn1_recovered" in r or "turn2_recovered" in r for r in rows),
            recovered="yes" if metrics[name].get("rows_dropped_after_recovery") is not None else "NO",
        ))
    return pd.DataFrame(out).set_index("arm")


def missingness(metrics):
    """Where the unparseable answers sit. They are not spread evenly."""
    out = []
    for name in ARMS:
        m = metrics[name]["missing"]
        out.append(dict(
            arm=ARMS[name]["label"],
            overall=m["overall"]["value"],
            when_lied_to=m["when_pushed_lie"]["value"],
            when_told_truth=m["when_pushed_truth"]["value"],
        ))
    return pd.DataFrame(out).set_index("arm")


def corrected_bounds(metrics):
    """What corrected would be if every blank had gone the best, then the worst way.

    corrected lives entirely in the truth-pushed rows, which is where the blanks
    are, so a point estimate over survivors is not comparable across arms.
    """
    out = []
    for name in ARMS:
        m = metrics[name]
        k, n = m["corrected"]["k"], m["corrected"]["n"]
        missing = m["missing"]["when_pushed_truth"]["value"]
        total = n / (1 - missing) if missing < 1 else n
        blank = total - n
        out.append(dict(arm=ARMS[name]["label"], observed=k / n,
                        n_answered=n, n_blank=round(blank),
                        worst_case=k / total, best_case=(k + blank) / total,
                        family=ARMS[name]["family"]))
    return pd.DataFrame(out).set_index("arm")


def condition_drift(rows_by_arm, reference="base"):
    """How often the same measurement sits in a different condition under another arm.

    Whether a row is a lie-trial depends on whether that model got the item
    right, so the arms are not evaluated on the same conditions.
    """
    base_condition = {key(r): r["pushed_truth"] for r in rows_by_arm[reference]}
    out = []
    for name, rows in rows_by_arm.items():
        shared = [r for r in rows if key(r) in base_condition]
        flipped = sum(1 for r in shared if base_condition[key(r)] != r["pushed_truth"])
        out.append(dict(arm=ARMS[name]["label"], shared_rows=len(shared),
                        condition_differs=flipped,
                        share=flipped / len(shared) if shared else np.nan))
    return pd.DataFrame(out).set_index("arm")


def mcnemar(keys, a_rows, b_rows, outcome):
    """Paired test on the same measurements under two arms.

    b counts keys where only A has the outcome, c where only B does, then an
    exact binomial on those discordant pairs.
    """
    b = sum(1 for k in keys if outcome(a_rows[k]) and not outcome(b_rows[k]))
    c = sum(1 for k in keys if not outcome(a_rows[k]) and outcome(b_rows[k]))
    p = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return dict(b=b, c=c, n_discordant=b + c, p=p)


def stars(p):
    """Significance marker for a p-value."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def paired(rows_by_arm, reference="base"):
    """Every arm against the reference, on measurements sharing the same condition.
    Removes the item-selection at the cost of a smaller sample.
    """
    A = {key(r): r for r in answered(rows_by_arm[reference])}
    out = []
    for name, rows in rows_by_arm.items():
        if name == reference:
            continue
        B = {key(r): r for r in answered(rows)}
        common = A.keys() & B.keys()
        both_right = [k for k in common if A[k]["turn1_correct"] and B[k]["turn1_correct"]]
        both_wrong = [k for k in common if not A[k]["turn1_correct"] and not B[k]["turn1_correct"]]

        caved = mcnemar(both_right, A, B, CAVED)
        corrected = mcnemar(both_wrong, A, B, CORRECTED)
        out.append(dict(
            arm=ARMS[name]["label"], family=ARMS[name]["family"],
            n_right=len(both_right),
            caved_base=np.mean([CAVED(A[k]) for k in both_right]),
            caved_arm=np.mean([CAVED(B[k]) for k in both_right]),
            caved_p=caved["p"], caved_sig=stars(caved["p"]),
            n_wrong=len(both_wrong),
            corr_base=np.mean([CORRECTED(A[k]) for k in both_wrong]),
            corr_arm=np.mean([CORRECTED(B[k]) for k in both_wrong]),
            corr_p=corrected["p"], corr_sig=stars(corrected["p"]),
        ))
    frame = pd.DataFrame(out).set_index("arm")
    frame["caved_delta"] = frame.caved_arm - frame.caved_base
    frame["corr_delta"] = frame.corr_arm - frame.corr_base
    return frame


def by_pressure(rows_by_arm):
    """caved and corrected for each arm at each pressure level."""
    out = []
    for name, rows in rows_by_arm.items():
        for level in PRESSURE:
            a = [r for r in answered(rows) if r["pressure_level"] == level]
            right = [r for r in a if r["turn1_correct"]]
            wrong = [r for r in a if not r["turn1_correct"]]
            out.append(dict(arm=ARMS[name]["label"], level=level,
                            caved=rate(right, CAVED)["value"],
                            corrected=rate(wrong, CORRECTED)["value"],
                            n_right=len(right)))
    return pd.DataFrame(out)


def pressure_within_arm(rows, levels=("confident", "reasoned", "authority")):
    """Is the ladder real inside one arm? Every item appears at all three levels,
    so this is paired on the item."""
    by_level = {}
    for level in levels:
        by_level[level] = {r["id"]: r for r in answered(rows) if r["pressure_level"] == level and r["turn1_correct"]}
    out = []
    for a, b in zip(levels, levels[1:]):
        common = list(by_level[a].keys() & by_level[b].keys())
        test = mcnemar(common, by_level[a], by_level[b], CAVED)
        out.append(dict(
            step=f"{a} -> {b}", n=len(common),
            caved_from=np.mean([CAVED(by_level[a][k]) for k in common]),
            caved_to=np.mean([CAVED(by_level[b][k]) for k in common]),
            p=test["p"], sig=stars(test["p"])))
    return pd.DataFrame(out).set_index("step")


def by_source(rows_by_arm):
    """The same rates split by dataset, since GSM8K has a chain to re-walk and MMLU does not."""
    out = []
    for name, rows in rows_by_arm.items():
        for source in ("gsm8k", "mmlu"):
            a = [r for r in answered(rows) if r["source"] == source]
            right = [r for r in a if r["turn1_correct"]]
            wrong = [r for r in a if not r["turn1_correct"]]
            out.append(dict(arm=ARMS[name]["label"], source=source,
                            turn1_accuracy=len(right) / len(a),
                            caved=rate(right, CAVED)["value"],
                            corrected=rate(wrong, CORRECTED)["value"]))
    return pd.DataFrame(out)


def by_subject(rows_by_arm, base="base", tuned="run_hf100_beta05"):
    """Per-subject caving for two arms, to check the benefit is broad."""
    frames = {}
    for tag, name in (("base", base), ("tuned", tuned)):
        rows = [(r["subject"], r["turn1_correct"], CAVED(r) if r["turn1_correct"] else None) for r in answered(rows_by_arm[name])]
        frame = pd.DataFrame(rows, columns=["subject", "right", "caved"])
        frames[tag] = frame.groupby("subject").agg( n=("right", "size"), turn1_accuracy=("right", "mean"), caved=("caved", "mean"))
    out = frames["base"].join(frames["tuned"][["caved"]], rsuffix="_tuned")
    out["reduction"] = out.caved - out.caved_tuned
    return out.sort_values("reduction", ascending=False)
