"""Datasets A and B with candidates from the same two query types (Table 5
and Section 7.2).

Qwen with the initial prompt (v0) and without reasoning tokens, by majority
vote over three runs. Dataset A has candidates from the headword and JMdict
queries only; on Dataset B, the candidates of those two query types were
judged on their own (02_judge.py --query-types headword jmdict), and the
development and test sets are combined and split into the Nature division
and the other divisions. For each group the script prints the records, the
EQ pairs, the recall, the false-positive rate (the share of the other pairs
predicted EQUAL), the F1 and the share of EQ pairs.

Then, for Dataset A and all of Dataset B, the share of EQ pairs with a
surface match (src/wlsp_bn/cells.py), and the recall among them.

    uv run python scripts/analysis/dataset_comparison.py
"""

import pandas as pd

from wlsp_bn import paths
from wlsp_bn.cells import cell_field, surface_match
from wlsp_bn.metrics import precision_recall_f1
from wlsp_bn.predictions import KEY, load_runs, majority

MODEL, PROMPT = "qwen3.5-27b", "v0"


def judged(data: str, dataset: str) -> pd.DataFrame:
    """The pairs of a condition with their majority vote and the fields used here."""
    pairs, runs = load_runs(paths.PREDICTIONS / data / MODEL / PROMPT)
    pairs["predicted"] = majority(runs)
    texts = pd.read_parquet(
        paths.INTERIM / f"pairs_{dataset}.parquet",
        columns=[*KEY, "wlsp_full", "bn_1step"],
    )
    pairs = pairs.merge(texts, on=KEY, how="left", validate="one_to_one")
    pairs["division"] = pairs["wlsp_full"].map(lambda t: cell_field(t, "部門"))
    pairs["match"] = [
        surface_match(w, b)
        for w, b in zip(pairs["wlsp_full"], pairs["bn_1step"], strict=True)
    ]
    return pairs


def main() -> None:
    a = judged("a", "a")
    b = pd.concat(
        [judged(f"b-{split}-headword-jmdict", "b") for split in ["dev", "test"]],
        ignore_index=True,
    )
    groups = {
        "A": a,
        "B (Nature)": b[b["division"] == "自然"],
        "B (Other divisions)": b[b["division"] != "自然"],
    }

    header = f"{'Records':>8}{'EQ':>6}{'Recall':>8}{'FP rate':>9}{'F1':>7}"
    print(f"{'':22}{header}  EQ rate")
    for name, g in groups.items():
        eq = g["label"] == "EQ"
        _, recall, f1 = precision_recall_f1(eq.tolist(), g["predicted"].tolist())
        fp_rate = g.loc[~eq, "predicted"].mean()
        print(
            f"{name:22}{g['record_id'].nunique():8}{eq.sum():6}{recall:8.3f}"
            f"{fp_rate:9.3f}{f1:7.3f}  {eq.mean():.1%}"
        )

    print("\nEQ pairs with a surface match, and the recall among them")
    for name, g in {"A": a, "B": b}.items():
        eq = g[g["label"] == "EQ"]
        matched = eq[eq["match"]]
        print(
            f"  {name}: {len(matched)}/{len(eq)} = {len(matched) / len(eq):.1%}, "
            f"recall {matched['predicted'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
