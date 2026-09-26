"""
The ORPO training curves, read from each run's train_log.json.
"""

import json

import pandas as pd

from episteme import config

# The 50/50 mix was stopped at step 48 once its preference term stayed flat, so
# it has no eval and only the opening of its curve.
MIXED_RUN = "run_beta05_lr2e5"
HOLD_FIRM_RUN = "run_hf100_beta05"

CURVE_COLUMNS = ["epoch", "log_odds_ratio", "rewards_accuracy", "log_odds_chosen", "nll_loss"]


def load_curve(run):
    """One run's logged metrics as a frame, one row per logging step."""
    log = json.loads((config.RESULTS / run / "train_log.json").read_text())
    return pd.DataFrame(log["log"], columns=log["columns"])


def curves():
    """The mixed and hold-firm runs as frames."""
    return load_curve(MIXED_RUN), load_curve(HOLD_FIRM_RUN)


def curve_comparison(mixed, hold_firm, until=0.58):
    """Mean of each metric over the window where both runs have a record."""
    return pd.DataFrame({
        "50/50 mix": mixed[mixed.epoch <= until][CURVE_COLUMNS[1:]].mean(),
        "100% hold-firm": hold_firm[hold_firm.epoch <= until][CURVE_COLUMNS[1:]].mean(),
    }).T


def curve_endpoints(hold_firm):
    """Start, end of the first pass, and end of training."""
    return pd.DataFrame({
        "start (epoch 0.06)": hold_firm.iloc[0][CURVE_COLUMNS[1:]],
        "end of pass 1 (1.08)": hold_firm[hold_firm.epoch <= 1.08].iloc[-1][CURVE_COLUMNS[1:]],
        "end of training (2.96)": hold_firm.iloc[-1][CURVE_COLUMNS[1:]],
    }).T
