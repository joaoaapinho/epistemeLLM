"""Confidence intervals for the rates in metrics.py."""

import numpy as np
from scipy import stats


def wilson_ci(successes, total, alpha=0.05):
    """
    Wilson interval for a proportion.

    Used instead of the textbook normal approximation because our groups are
    small and some rates sit near 0 or 1, where that approximation returns
    nonsense like a negative lower bound.
    """
    if total == 0:
        return None, None

    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
