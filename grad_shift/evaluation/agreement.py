"""Inter-judge agreement statistics shared by the judge calibration scripts and
the paper's camera-ready judge validation (``paper/products/judge_validation``).
"""

import itertools

import numpy as np


def krippendorff_alpha_interval(ratings: np.ndarray) -> float:
    """Krippendorff's alpha for interval data, items x raters, no missing cells.

    alpha = 1 - D_o/D_e, where D_o is the mean squared difference between raters
    within an item and D_e is the mean squared difference between all ratings
    regardless of item. Properties (asserted in tests rather than trusting a
    remembered reference value): identical raters give 1.0, and raters that are
    independent draws from the same marginal give approximately 0.0.
    """
    ratings = np.asarray(ratings, dtype=float)
    n_items, n_raters = ratings.shape
    if n_raters < 2:
        raise ValueError("need >= 2 raters")

    # Observed disagreement: within-item pairwise squared differences.
    within = 0.0
    for j, k in itertools.combinations(range(n_raters), 2):
        within += np.sum((ratings[:, j] - ratings[:, k]) ** 2)
    n_within_pairs = n_items * n_raters * (n_raters - 1) / 2
    d_o = within / n_within_pairs

    # Expected disagreement: pairwise squared differences over the pooled values.
    flat = ratings.ravel()
    total = len(flat)
    # sum_{i<j}(x_i-x_j)^2 == n*sum(x^2) - (sum x)^2, avoiding an O(n^2) loop.
    sq = total * np.sum(flat ** 2) - np.sum(flat) ** 2
    d_e = sq / (total * (total - 1) / 2)

    if d_e == 0:
        return float("nan")
    return 1.0 - d_o / d_e
