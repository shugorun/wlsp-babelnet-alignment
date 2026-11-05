"""Scoring binary EQ predictions, with confidence intervals by record.

EQ pairs are the positive class. A record can have several EQ pairs, and
the predictions for the pairs of one record are not independent, so the
bootstrap resamples records, not pairs.
"""

from collections.abc import Hashable, Sequence

import numpy as np

# Every bootstrap uses this seed, so that two comparisons over the same
# records resample them alike.
BOOTSTRAP_SEED = 20260713
BOOTSTRAP_REPLICATES = 10_000


def precision_recall_f1(
    y_true: Sequence[bool], y_pred: Sequence[bool]
) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class; 0.0 where undefined."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mean_f1_diff_ci(
    record_ids: Sequence[Hashable],
    y_true: Sequence[bool],
    runs_a: Sequence[Sequence[bool]],
    runs_b: Sequence[Sequence[bool]],
) -> tuple[float, float, float]:
    """The difference in F1 between two conditions, with its 95% interval.

    Each condition is a list of runs over the same pairs. The F1 of a
    condition is the mean of the F1 of its runs, and the difference is that
    of A minus that of B. Each bootstrap replicate draws the records once
    and uses them for every run of both conditions.

    Returns
    -------
    tuple of float
        The difference, and the 2.5th and 97.5th percentiles of its
        bootstrap distribution.
    """
    for run in [*runs_a, *runs_b]:
        if not len(run) == len(y_true) == len(record_ids):
            raise ValueError("a run does not cover the same pairs")

    def mean_f1(y: Sequence[bool], runs: Sequence[Sequence[bool]]) -> float:
        return float(np.mean([precision_recall_f1(y, run)[2] for run in runs]))

    groups: dict[Hashable, dict] = {}
    for i, rid in enumerate(record_ids):
        g = groups.setdefault(
            rid, {"y": [], "a": [[] for _ in runs_a], "b": [[] for _ in runs_b]}
        )
        g["y"].append(y_true[i])
        for j, run in enumerate(runs_a):
            g["a"][j].append(run[i])
        for j, run in enumerate(runs_b):
            g["b"][j].append(run[i])
    rids = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    point = mean_f1(y_true, runs_a) - mean_f1(y_true, runs_b)

    diffs = np.empty(BOOTSTRAP_REPLICATES)
    for b in range(BOOTSTRAP_REPLICATES):
        y: list[bool] = []
        a: list[list[bool]] = [[] for _ in runs_a]
        bb: list[list[bool]] = [[] for _ in runs_b]
        for idx in rng.choice(len(rids), size=len(rids), replace=True):
            g = groups[rids[idx]]
            y.extend(g["y"])
            for j in range(len(runs_a)):
                a[j].extend(g["a"][j])
            for j in range(len(runs_b)):
                bb[j].extend(g["b"][j])
        diffs[b] = mean_f1(y, a) - mean_f1(y, bb)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)
