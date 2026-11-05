"""The false-positive rates of Qwen on Dataset B (Table 4, top, and Section 7.1).

For the optimized (v9b) and initial (v0) prompts without reasoning tokens,
on the development and the test set: the share of the BR, NR and OT pairs
that a run predicts EQUAL, averaged over the runs of the condition. Then,
on the test set, the same share among the OT pairs with and without a
surface match: the WLSP headword or a co-classified word is a Japanese
lemma of the synset (src/wlsp_bn/cells.py).

    uv run python scripts/analysis/false_positives.py
"""

import statistics

import pandas as pd

from wlsp_bn import paths
from wlsp_bn.cells import surface_match
from wlsp_bn.predictions import KEY, load_runs

PROMPTS = {"optimized": "v9b", "initial": "v0"}
SPLITS = ["dev", "test"]
LABELS = ["BR", "NR", "OT"]


def rate(runs: list[list[bool]], at: pd.Series) -> float:
    """The share of the pairs `at` predicted EQUAL, averaged over the runs."""
    return statistics.mean(pd.Series(run)[at.to_numpy()].mean() for run in runs)


def main() -> None:
    texts = pd.read_parquet(
        paths.INTERIM / "pairs_b.parquet", columns=[*KEY, "wlsp_full", "bn_1step"]
    )
    loaded = {
        (name, split): load_runs(
            paths.PREDICTIONS / f"b-{split}" / "qwen3.5-27b" / prompt
        )
        for name, prompt in PROMPTS.items()
        for split in SPLITS
    }

    print("False-positive rate by gold label (mean over runs)")
    print(f"{'':6}" + "".join(f"{f'{n} {s}':>16}" for n, s in loaded))
    for label in LABELS:
        rates = [rate(runs, pairs["label"] == label) for pairs, runs in loaded.values()]
        print(f"{label:6}" + "".join(f"{r:16.3f}" for r in rates))
    print("runs  " + "".join(f"{len(runs):16}" for _, runs in loaded.values()))

    print("\nOT pairs of the test set, with / without a surface match")
    for name in PROMPTS:
        pairs, runs = loaded[name, "test"]
        pairs = pairs.merge(texts, on=KEY, how="left", validate="one_to_one")
        match = pd.Series(
            [
                surface_match(w, b)
                for w, b in zip(pairs["wlsp_full"], pairs["bn_1step"], strict=True)
            ]
        )
        ot = pairs["label"] == "OT"
        with_match, without = rate(runs, ot & match), rate(runs, ot & ~match)
        print(
            f"  {name}: {with_match:.3f} ({(ot & match).sum()} pairs) / "
            f"{without:.3f} ({(ot & ~match).sum()} pairs)"
        )


if __name__ == "__main__":
    main()
