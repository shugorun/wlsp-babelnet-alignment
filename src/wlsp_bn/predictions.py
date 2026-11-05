"""The runs of a condition, as scripts/evaluation/02_judge.py and the
baselines write them, with the labels of the pairs they cover.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from wlsp_bn import paths

KEY = ["record_id", "synset_id"]


def load_runs(condition: Path) -> tuple[pd.DataFrame, list[list[bool]]]:
    """The pairs of a condition with their labels, and each run's EQUAL predictions.

    `condition` is a directory under data/predictions/, whose first part
    names the dataset whose labels in data/annotations/ are used. The runs
    are in the order of their numbers, and each covers the pairs of the
    first run, in the same order.
    """
    condition = condition.resolve()
    data = condition.relative_to(paths.PREDICTIONS).parts[0]
    gold = pd.read_csv(paths.ANNOTATIONS / f"dataset_{data[0]}.tsv", sep="\t")
    files = sorted(condition.glob("run*.tsv"), key=lambda p: int(p.stem[3:]))
    if not files:
        raise SystemExit(f"{condition} has no runs")
    first = pd.read_csv(files[0], sep="\t")
    pairs = first[KEY].merge(gold, on=KEY, how="left", validate="one_to_one")
    if pairs["label"].isna().any():
        raise SystemExit(f"{files[0]} has pairs that are not annotated")
    runs = []
    for f in files:
        run = pd.read_csv(f, sep="\t")
        if not run[KEY].equals(first[KEY]):
            raise SystemExit(f"{f} does not cover the pairs of {files[0]}")
        runs.append((run["prediction"] == "EQUAL").tolist())
    return pairs, runs


def majority(runs: Sequence[Sequence[bool]]) -> list[bool]:
    """Whether more than half of the runs predict each pair EQUAL."""
    return [2 * sum(votes) > len(runs) for votes in zip(*runs, strict=True)]
