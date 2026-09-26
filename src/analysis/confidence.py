"""
Caving against the model's own turn-1 confidence, and the headroom that implies.
"""

import numpy as np
import pandas as pd
from scipy import stats

from episteme.analysis.registry import ARMS, CAVED, answered
from episteme.intervals import wilson_ci

# Fixed abs bins so the arms are read on the same x-axis. Quantile bins
# would hide that the two models have different confidence distributions.
CONF_BINS = [-0.60, -0.30, -0.20, -0.15, -0.11, -0.08, 0.0]


def confidence_frame(rows):
    """One row per measurement the model got right at turn 1."""
    kept = [r for r in answered(rows) if r["turn1_correct"] and r["turn1_confidence"] is not None]
    return pd.DataFrame(dict(
        conf=[r["turn1_confidence"] for r in kept],
        caved=[float(CAVED(r)) for r in kept],
        level=[r["pressure_level"] for r in kept],
        source=[r["source"] for r in kept]))


def confidence_bins(rows, edges=CONF_BINS):
    """Caved rate per confidence bin, with intervals and bin centres to plot on."""
    frame = confidence_frame(rows)
    grouped = frame.groupby(pd.cut(frame.conf, edges), observed=True).caved.agg(["mean", "count"])
    grouped["lo"], grouped["hi"] = zip(*[wilson_ci(int(m * c), int(c)) for m, c in zip(grouped["mean"], grouped["count"])])
    grouped["centre"] = [(i.left + i.right) / 2 for i in grouped.index]
    return grouped


def confidence_fit(rows_by_arm):
    """Logistic fit of P(caved) on the model's own turn-1 log-probability."""
    import statsmodels.api as sm
    out = []
    for name, rows in rows_by_arm.items():
        frame = confidence_frame(rows)
        model = sm.Logit(frame.caved.values, sm.add_constant(frame.conf.values)).fit(disp=0)
        out.append(dict(arm=ARMS[name]["label"], n=len(frame), mean_conf=frame.conf.mean(), slope=model.params[1], p=model.pvalues[1], pseudo_r2=model.prsquared))
    return pd.DataFrame(out).set_index("arm")


def confidence_gate(rows, n_thresholds=801):
    """What "keep your answer above threshold t" would score, on confidence alone.

    Right rows had a lie pushed, so keeping is correct and yielding is a cave.
    Wrong rows had the truth pushed, so yielding is the correction.
    """
    kept = [r for r in answered(rows) if r["turn1_confidence"] is not None]
    conf = np.array([r["turn1_confidence"] for r in kept])
    right = np.array([r["turn1_correct"] for r in kept], bool)
    thresholds = np.quantile(conf, np.linspace(0.01, 0.99, n_thresholds))
    return pd.DataFrame(dict(
        threshold=thresholds,
        caved=[(conf[right] < t).mean() for t in thresholds],
        corrected=[(conf[~right] < t).mean() for t in thresholds])
        )


def confidence_auc(rows):
    """How well the model's own confidence separates its right answers from its wrong ones."""
    kept = [r for r in answered(rows) if r["turn1_confidence"] is not None]
    conf = np.array([r["turn1_confidence"] for r in kept])
    right = np.array([r["turn1_correct"] for r in kept], bool)
    u = stats.mannwhitneyu(conf[right], conf[~right], alternative="greater")
    return u.statistic / (right.sum() * (~right).sum())
