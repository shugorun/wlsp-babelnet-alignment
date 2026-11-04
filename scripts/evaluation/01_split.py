"""Split the records of Dataset B into a development and a test set.

Reads data/interim/pairs_b.parquet and writes data/splits/dataset_b.tsv:
record_id and split (dev or test), 100 records each.

The split is stratified: within each WLSP division, the records with 0, 1,
2 and 3 or more EQ pairs are each split in half, and the divisions get the
numbers of dev and test records in DIVISIONS. Each seed from 0 to 9,999
gives one such split, and the most balanced one is kept: the one with the
smallest sum, over the counts in BALANCED, of the difference between dev
and test divided by the total. A surface match is a pair whose synset has
the record's headword or a co-classified word among its Japanese lemmas
(src/wlsp_bn/cells.py).

    uv run python scripts/evaluation/01_split.py
"""

import random

import pandas as pd

from wlsp_bn import paths
from wlsp_bn.cells import cell_field, surface_match

PAIRS = paths.INTERIM / "pairs_b.parquet"
OUT = paths.SPLITS / "dataset_b.tsv"

# The (dev, test) numbers of records of each division, in the order the
# seeds draw them. Activities has one more record in test, Nature one more
# in dev.
DIVISIONS = {
    "関係": (19, 19),  # Relations
    "主体": (15, 15),  # Agents
    "活動": (38, 39),  # Activities
    "生産物": (14, 14),  # Products
    "自然": (14, 13),  # Nature
}
SEEDS = range(10_000)
BALANCED = ["eq", "eq_match", "other_match", "pairs", "nr", "br"]


def record_table(pairs: pd.DataFrame) -> list[dict]:
    """The division and counts of each record, in the order of the pairs."""
    table = []
    for record_id, g in pairs.groupby("record_id", sort=False):
        eq = g["label"] == "EQ"
        texts = zip(g["wlsp_full"], g["bn_1step"], strict=True)
        match = pd.Series([surface_match(w, b) for w, b in texts], index=g.index)
        table.append(
            {
                "record_id": int(record_id),
                "division": cell_field(g["wlsp_full"].iloc[0], "部門"),
                "pairs": len(g),
                "eq": int(eq.sum()),
                "eq_match": int((eq & match).sum()),
                "other_match": int((~eq & match).sum()),
                "nr": int((g["label"] == "NR").sum()),
                "br": int((g["label"] == "BR").sum()),
            }
        )
    return table


def dev_records(table: list[dict], seed: int) -> list[int]:
    """The dev records of the stratified split that `seed` gives."""
    rng = random.Random(seed)
    dev = []
    for division, (n_dev, _) in DIVISIONS.items():
        records = [r for r in table if r["division"] == division]
        stratum = {r["record_id"]: min(r["eq"], 3) for r in records}
        picked = []
        # A stratum with an odd number of records gives the extra one to
        # dev and to test in turn.
        extra = 0
        for s in sorted(set(stratum.values())):
            ids = [i for i in stratum if stratum[i] == s]
            rng.shuffle(ids)
            k = len(ids) // 2
            if len(ids) % 2 == 1:
                k += extra
                extra = 1 - extra
            picked += ids[:k]
        # The division's number of dev records is then reached by moving
        # records without EQ pairs.
        outside = [i for i in stratum if stratum[i] == 0 and i not in picked]
        inside = [i for i in picked if stratum[i] == 0]
        while len(picked) < n_dev and outside:
            picked.append(outside.pop())
        while len(picked) > n_dev and inside:
            picked.remove(inside.pop())
        if len(picked) != n_dev:
            raise SystemExit(f"{division}: {len(picked)} dev records, not {n_dev}")
        dev += picked
    return dev


def imbalance(table: list[dict], dev: set[int]) -> float:
    score = 0.0
    for key in BALANCED:
        total = sum(r[key] for r in table)
        in_dev = sum(r[key] for r in table if r["record_id"] in dev)
        score += abs(in_dev - (total - in_dev)) / max(1, total)
    return score


def main() -> None:
    pairs = pd.read_parquet(PAIRS)
    table = record_table(pairs)
    unknown = {r["division"] for r in table} - set(DIVISIONS)
    if unknown:
        raise SystemExit(f"unknown divisions: {unknown}")

    best = min(SEEDS, key=lambda s: imbalance(table, set(dev_records(table, s))))
    dev = set(dev_records(table, best))
    ids = sorted(r["record_id"] for r in table)
    split = pd.DataFrame(
        {"record_id": ids, "split": ["dev" if i in dev else "test" for i in ids]}
    )
    paths.SPLITS.mkdir(parents=True, exist_ok=True)
    split.to_csv(OUT, sep="\t", index=False, lineterminator="\n")

    print(f"seed {best}: imbalance {imbalance(table, dev):.6f}")
    for key in BALANCED:
        in_dev = sum(r[key] for r in table if r["record_id"] in dev)
        in_test = sum(r[key] for r in table if r["record_id"] not in dev)
        print(f"  {key}: dev {in_dev}, test {in_test}")
    print(f"{OUT}: {len(dev)} dev and {len(table) - len(dev)} test records")


if __name__ == "__main__":
    main()
