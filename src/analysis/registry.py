"""
Arm vocabulary, result loading, and the row helpers everything else shares.
"""

import json

import numpy as np

from episteme import config
from episteme.intervals import wilson_ci

# The seven arms in the order every table and chart uses them. family drives
# colour: blue for the control, red for prompt engineering, green for fine-tuned.
ARMS = {
    "base": dict(label="base (control)", family="control", kind="prompt"),
    "prompt_minimal": dict(label="minimal", family="prompt", kind="prompt"),
    "prompt_resist": dict(label="resist", family="prompt", kind="prompt"),
    "prompt_specific": dict(label="specific", family="prompt", kind="prompt"),
    "prompt_verify": dict(label="verify", family="prompt", kind="prompt"),
    "run_50_50": dict(label="ORPO 50/50", family="tuned", kind="tuned"),
    "run_hf100_beta05": dict(label="ORPO 100% hold-firm", family="tuned", kind="tuned"),
}
LABELS = {name: meta["label"] for name, meta in ARMS.items()}
BY_LABEL = {meta["label"]: name for name, meta in ARMS.items()}

PRESSURE = ("confident", "reasoned", "authority")

# Outcome predicates, matching src/metrics.py.
CAVED = lambda r: not r["turn2_correct"] # among rows right at turn 1
CORRECTED = lambda r: r["turn2_correct"] # among rows wrong at turn 1
CHANGED = lambda r: r["changed_mind"]


def load(name):
    """Every measurement one arm recorded."""
    path = config.RESULTS / name / "responses.jsonl"
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def load_meta(name, filename):
    """One of an arm's sidecar json files, or {} if the run never wrote it."""
    path = config.RESULTS / name / filename
    return json.loads(path.read_text()) if path.exists() else {}


def load_all():
    """rows, metrics.json, run_info.json and train_info.json for every arm."""
    rows = {name: load(name) for name in ARMS}
    metrics = {name: load_meta(name, "metrics.json") for name in ARMS}
    info = {name: load_meta(name, "run_info.json") for name in ARMS}
    train = {name: load_meta(name, "train_info.json") for name in ARMS}
    return rows, metrics, info, train


def key(row):
    """A measurement is identified by the item and the pressure level."""
    return (row["id"], row["pressure_level"])


def answered(rows):
    """Only rows the model gave a readable answer to in both turns."""
    return [r for r in rows if r["answered_both_turns"]]


def buckets(rows):
    """The answered rows split by whether turn 1 was right."""
    a = answered(rows)
    return a, [r for r in a if r["turn1_correct"]], [r for r in a if not r["turn1_correct"]]


def rate(rows, condition):
    """Share of rows satisfying the condition, with a Wilson interval."""
    n = len(rows)
    k = sum(1 for r in rows if condition(r))
    low, high = wilson_ci(k, n)
    return dict(value=k / n if n else np.nan, k=k, n=n, lo=low, hi=high)
