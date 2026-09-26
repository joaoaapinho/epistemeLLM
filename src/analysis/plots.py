"""
Every analysis chart.
"""

import importlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from episteme import config
from episteme.analysis.registry import ARMS, BY_LABEL, PRESSURE

# Color scheme.
C = dict(blue="#89b4fa", green="#a6e3a1", red="#f38ba8", peach="#fab387",
         mauve="#cba6f7", yellow="#f9e2af", teal="#94e2d5", lavender="#b4befe")

SURFACE = "#1e1e2e" # base - also ring colour between overlapping marks
INK = "#cdd6f4" # text
INK2 = "#a6adc8" # subtext0
MUTED = "#6c7086" # overlay0
GRID = "#313244" # surface0

FAMILY = {"control": C["blue"], "prompt": C["red"], "tuned": C["green"]}

# The only remaining Mocha pair clearing the chroma floor, for non-arm series.
NON_ARM = [C["mauve"], C["peach"]]

# The pressure levels are ordered, so they get an ordinal ramp - but in each
# arm's own family hue, so a bar says both which arm it is and which level.
# Validated per ramp: monotone lightness, adjacent dL >= 0.06, dark end >= 3:1
# against the surface, hue spread under 9 degrees.
LADDER = {
    "control": ["#5c75a4", "#89b4fa", "#bed6fc"],
    "prompt":  ["#9a5d75", "#f38ba8", "#f8bfcf"],
    "tuned":   ["#6d9071", "#a6e3a1", "#cef0cb"],
}

FIGS = config.ROOT/"report"/"figures"


def setup():
    """Apply the Mocha stylesheet, then override structure but never colour."""
    core = importlib.import_module("matplotlib.style.core")
    for new, old in (("_read_style_directory", "read_style_directory"),
                     ("_update_nested_dict", "update_nested_dict")):
        if not hasattr(core, old):
            setattr(core, old, getattr(mpl.style, new))
    from catppuccin.extras.matplotlib import CATPPUCCIN_STYLE_DIRECTORY

    plt.style.use(CATPPUCCIN_STYLE_DIRECTORY / "mocha.mplstyle")
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.labelpad": 8,
        "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
    })
    FIGS.mkdir(parents=True, exist_ok=True)
    return CATPPUCCIN_STYLE_DIRECTORY / "mocha.mplstyle"


TITLE_SIZE = 12.5
SUBTITLE_SIZE = 9.5


def _tidy(ax, grid="y"):
    """Drop the top/right spines and put gridlines only where they help read values.

    grid is "x", "y" or "both" - bars get one axis, scatters get both.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for axis in ("x", "y"):
        ax.grid(axis=axis, visible=grid in (axis, "both"), alpha=0.55)
    return ax


def _figtitle(fig, title, subtitle=None):
    """Flush to the figure's left edge, above everything, with room to breathe."""
    fig.text(0.008, 1.075 if subtitle else 1.02, title, ha="left", va="bottom",
             fontsize=TITLE_SIZE, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.008, 1.015, subtitle, ha="left", va="bottom",
                 fontsize=SUBTITLE_SIZE, color=MUTED)


def _title(ax, title, subtitle=None):
    """What the chart is. The insight belongs in the surrounding text, not here.

    x is flush to the figure's left edge, y sits a fixed distance above the plot,
    so the gap looks the same whatever the figure height.
    """
    def place(text, offset, size, colour):
        ax.annotate(text, xy=(0, 1), xycoords=("figure fraction", "axes fraction"),
                    xytext=(7, offset), textcoords="offset points", ha="left",
                    va="bottom", fontsize=size, color=colour,
                    fontweight="bold" if colour == INK else "normal")

    place(title, 26 if subtitle else 12, TITLE_SIZE, INK)
    if subtitle:
        place(subtitle, 11, SUBTITLE_SIZE, MUTED)


def _save(fig, name):
    """Write the figure to report/figures and show it."""
    fig.savefig(FIGS / f"{name}.png")
    plt.show()


def _colours(labels):
    """Family colour for each arm label."""
    return [FAMILY[ARMS[BY_LABEL[l]]["family"]] for l in labels]


# Two-series charts take the dark and light ends; three take the whole ramp.
STEPS = {2: [0, 2], 3: [0, 1, 2]}
GREYS = ["#585868", "#8b8b9c", "#c3c3ce"]


def _shades(labels, i, n):
    """Shade i of n from each arm's own family ramp."""
    step = STEPS[n][i]
    return [LADDER[ARMS[BY_LABEL[l]]["family"]][step] for l in labels]


def _shade_legend(ax, names, **kw):
    """Neutral swatches, since hue is already carrying arm identity."""
    greys = [GREYS[i] for i in STEPS[len(names)]]
    ax.legend(handles=[Patch(facecolor=g, label=n) for g, n in zip(greys, names)],
              handlelength=1.6, handleheight=1.1, **kw)


def missing(frame):
    """Blanks by condition. Two conditions, not two arms, so NON_ARM."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y, h = np.arange(len(frame)), 0.36
    ax.barh(y + h / 2, frame.when_lied_to * 100, height=h,
            color=_shades(frame.index, 0, 2))
    ax.barh(y - h / 2, frame.when_told_truth * 100, height=h,
            color=_shades(frame.index, 1, 2))
    ax.set_yticks(y); ax.set_yticklabels(frame.index)
    ax.set_xlabel("share of rows with no parseable answer (%)")
    _title(ax, "Unparseable answers, by condition",
           "dark to light within each arm's colour: pushed a lie, pushed the truth")
    _shade_legend(ax, ["pushed a lie (model was right)",
                       "pushed the truth (model was wrong)"], loc="lower right")
    _tidy(ax, grid="x"); ax.invert_yaxis()
    _save(fig, "missingness")


def frontier(head):
    """caved against corrected. The ideal corner is bottom-right."""
    nudge = {
        "base": (0.006, 0.013, "left"), "run_50_50": (-0.006, 0.011, "right"),
        "prompt_resist": (-0.006, -0.004, "right"),
        "prompt_specific": (0.006, 0.007, "left"),
        "prompt_verify": (0.006, -0.009, "left"),
        "prompt_minimal": (-0.017, 0.0, "right"),
        "run_hf100_beta05": (0.036, 0.0, "left"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for name in ARMS:
        row = head.loc[ARMS[name]["label"]]
        colour = FAMILY[row.family]
        ax.errorbar(row.corrected, row.caved,
                    xerr=[[row.corrected - row.corrected_lo], [row.corrected_hi - row.corrected]],
                    yerr=[[row.caved - row.caved_lo], [row.caved_hi - row.caved]],
                    fmt="o", ms=9, color=colour, ecolor=colour, elinewidth=1.4,
                    capsize=0, mec=SURFACE, mew=1.4, zorder=3)
        dx, dy, ha = nudge[name]
        ax.annotate(ARMS[name]["label"], (row.corrected + dx, row.caved + dy),
                    fontsize=8.5, color=INK, ha=ha, va="center")
    ax.set_xlabel("corrected  →  accepts real corrections")
    ax.set_ylabel("caved  →  abandons correct answers")
    _title(ax, "Sycophancy against corrigibility",
           "all seven arms; bars are 95% Wilson intervals, bottom-right is better")
    ax.annotate("ideal", xy=(0.985, 0.06), xycoords="axes fraction", ha="right",
                fontsize=9, color=MUTED, style="italic")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=v, mec=SURFACE,
                                  mew=1.2, label=k) for k, v in FAMILY.items()],
              loc="upper left")
    _tidy(ax, grid="both")
    _save(fig, "frontier")


def reference_policies(ref):
    """Observed accuracy against always-comply and never-change."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y = np.arange(len(ref))
    ax.hlines(y, ref.if_always_complied, ref.if_never_changed, color=GRID, lw=6, zorder=1)
    ax.scatter(ref.if_never_changed, y, s=42, color=MUTED, zorder=3,
               label="if it never changed its mind")
    ax.scatter(ref.if_always_complied, y, s=42, color=MUTED, marker="s", zorder=3,
               label="if it always complied")
    # the gap the label measures, drawn so the number sits on what it describes
    for i, (v, hold) in enumerate(zip(ref.accuracy_after, ref.if_never_changed)):
        ax.plot([v, hold], [i, i], color=INK2, lw=1.4, ls=(0, (2.5, 2)), zorder=2)
        ax.annotate(f"{v - hold:+.3f}", ((v + hold) / 2, i), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=INK2)
    ax.scatter(ref.accuracy_after, y, s=90, zorder=4, color=_colours(ref.index),
               edgecolor=SURFACE, linewidth=1.4, label="observed")
    ax.set_yticks(y); ax.set_yticklabels(ref.index)
    ax.set_xlabel("final accuracy after pushback")
    _title(ax, "Final accuracy against two reference policies",
           "dashed span = how far observed accuracy falls short of never changing its mind")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.margins(y=0.10)
    _tidy(ax, grid="x"); ax.invert_yaxis()
    _save(fig, "reference_policies")


def bounds(frame):
    """corrected, with the range every blank could have moved it to."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(frame))
    ax.hlines(y, frame.worst_case, frame.best_case, color=GRID, lw=7, zorder=1)
    ax.scatter(frame.observed, y, s=80, zorder=3, color=_colours(frame.index),
               edgecolor=SURFACE, linewidth=1.3)
    ax.set_yticks(y); ax.set_yticklabels(frame.index)
    ax.set_xlabel("corrected")
    _title(ax, "Corrected, bounded by the missing answers",
           "dot = observed; bar = the range if every blank had gone one way")
    _tidy(ax, grid="x"); ax.invert_yaxis()
    _save(fig, "corrected_bounds")


def paired_shift(frame):
    """Where each arm moved from base, on identical measurements."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    y = np.arange(len(frame))
    for ax, (base_col, arm_col, title) in zip(axes, [
            ("caved_base", "caved_arm", "caved  (lower is better)"),
            ("corr_base", "corr_arm", "corrected  (higher is better)")]):
        ax.hlines(y, frame[base_col], frame[arm_col], color=GRID, lw=2.5, zorder=1)
        ax.scatter(frame[base_col], y, s=48, color=C["blue"], zorder=3,
                   edgecolor=SURFACE, linewidth=1.0)
        ax.scatter(frame[arm_col], y, s=64, zorder=4, edgecolor=SURFACE, linewidth=1.2,
                   color=[FAMILY[f] for f in frame.family])
        ax.set_yticks(y); ax.set_yticklabels(frame.index)
        ax.set_title(title, loc="left", pad=12, fontsize=10.5,
                     fontweight="bold", color=INK)
        _tidy(ax, grid="x"); ax.invert_yaxis()
    axes[0].legend(handles=[
        plt.Line2D([], [], marker="o", ls="", ms=7, color=C["blue"], label="base"),
        plt.Line2D([], [], marker="o", ls="", ms=7, color=C["red"], label="prompt arm"),
        plt.Line2D([], [], marker="o", ls="", ms=7, color=C["green"], label="fine-tuned arm"),
    ], loc="upper left")
    fig.tight_layout()
    _figtitle(fig, "Shift from base, paired",
              "restricted to measurements both arms answered in the same condition")
    _save(fig, "paired_shift")


def pressure(frame, order):
    """Caving by pressure type, each arm shaded in its own family hue."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    x, w = np.arange(len(order)), 0.26
    for i, level in enumerate(PRESSURE):
        sub = frame[frame.level == level].set_index("arm").loc[order]
        ax.bar(x + (i - 1) * w, sub.caved, width=w - 0.055,
               color=_shades(order, i, 3))
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("caved"); ax.set_ylim(0, 1.12)
    _title(ax, "Caving by type of pushback",
           "shaded dark to light within each arm's colour: confident, reasoned, authority")

    _shade_legend(ax, list(PRESSURE), ncol=3, loc="upper right")
    _tidy(ax)
    _save(fig, "pressure_levels")


def confidence(base_bins, tuned_bins, gate=None):
    """Caving against the model's own turn-1 confidence."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for bins, colour, label, nudge in ((base_bins, C["blue"], "base (control)", 12),
                                       (tuned_bins, C["green"], "ORPO 100% hold-firm", -17)):
        ax.plot(bins.centre, bins["mean"], "-o", color=colour, lw=2, ms=7,
                mec=SURFACE, mew=1.2, label=label, zorder=3)
        ax.fill_between(bins.centre, bins.lo, bins.hi, color=colour, alpha=0.20, lw=0, zorder=1)
        for x, m, n in zip(bins.centre, bins["mean"], bins["count"]):
            ax.annotate(f"n={int(n)}", (x, m), textcoords="offset points",
                        xytext=(0, nudge), ha="center", fontsize=7.5, color=MUTED)
    ax.set_xlabel("turn-1 mean token log-probability  →  more confident")
    ax.set_ylabel("caved"); ax.set_ylim(0, 1)
    _title(ax, "Caving by the model's own turn-1 confidence",
           "fixed bins on mean token log-probability; bands are 95% Wilson")
    ax.legend(loc="upper right")
    _tidy(ax, grid="both")
    _save(fig, "confidence")


def headroom(gate, points):
    """
    What the model's own confidence makes achievable, with the arms on top.
    points is a list of (label, corrected, caved, colour, (dx, dy)).
    Every arm sits above the curve, so the vertical distance is the part of
    the signal that arm did not use.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(gate.corrected, gate.caved, lw=2, color=C["mauve"], zorder=2,
            label="keep your answer above a confidence threshold")
    for label, corrected, caved, colour, offset in points:
        row = gate.iloc[(gate.corrected - corrected).abs().argmin()]
        ax.vlines(corrected, row.caved, caved, color=MUTED, lw=1,
                  ls=(0, (3, 3)), zorder=1)
        ax.annotate(f"{caved - row.caved:+.2f}", (corrected, (caved + row.caved) / 2),
                    textcoords="offset points", xytext=(-8, 0), ha="right",
                    fontsize=8, color=MUTED)
        ax.scatter([corrected], [caved], s=95, color=colour, edgecolor=SURFACE,
                   linewidth=1.4, zorder=4)
        ax.annotate(label, (corrected, caved), textcoords="offset points",
                    xytext=offset, fontsize=8.5, color=INK)
    ax.set_xlabel("corrected  →  accepts real corrections")
    ax.set_ylabel("caved  →  abandons correct answers")
    _title(ax, "Measured arms against a confidence-threshold policy",
           "the curve is what the model's own confidence alone would achieve")
    ax.legend(loc="upper left")
    _tidy(ax, grid="both")
    _save(fig, "headroom")


def degeneracy(frame):
    """Surface markers. Two phrasing categories, not arms, so NON_ARM."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x, w = np.arange(len(frame)), 0.38
    ax.bar(x - w / 2, frame.capitulation * 100, width=w - 0.055,
           color=_shades(frame.index, 0, 2))
    ax.bar(x + w / 2, frame.deference * 100, width=w - 0.055,
           color=_shades(frame.index, 1, 2))
    ax.set_xticks(x); ax.set_xticklabels(frame.index, rotation=20, ha="right")
    ax.set_ylabel("% of turn-2 replies")
    _title(ax, "Phrasing markers in turn-2 replies",
           "dark to light within each arm's colour: capitulation, deference")
    _shade_legend(ax, ["capitulation language", "deference language"])
    _tidy(ax)
    _save(fig, "degeneracy")


def training(hold_firm):
    """
    The first pass only. It is a complete record, and it is the pass where every
    pair is seen once, so the trend is generalisation rather than repetition.
    """
    first = hold_firm[hold_firm.epoch <= 1.08]
    panels = [
        ("log_odds_ratio", "log_odds_ratio",
         "−0.693 = chosen and rejected equally likely", -np.log(2)),
        ("rewards_accuracy", "rewards/accuracies", "0.5 = coin flip", 0.5),
        ("nll_loss", "nll_loss", "the supervised term, for contrast", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    for ax, (col, title, note, line) in zip(axes, panels):
        if line is not None:
            ax.axhline(line, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.annotate("no signal", (0.97, line), xycoords=("axes fraction", "data"),
                        fontsize=8, color=MUTED, va="bottom", ha="right")
        ax.plot(first.epoch, first[col], "-o", ms=4.5, lw=2, color=C["green"],
                mec=SURFACE, mew=0.9, zorder=3)
        ax.set_title(title, loc="left", pad=20, fontsize=10.5,
                     fontweight="bold", color=INK)
        ax.annotate(note, (0, 1.012), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=8.5, color=MUTED)
        ax.set_xlabel("epoch")
        _tidy(ax, grid="both")
    fig.tight_layout()
    _figtitle(fig, "ORPO 100% hold-firm, first pass over the data",
              "every pair seen exactly once; logged every 5 steps")
    _save(fig, "training_dynamics")
