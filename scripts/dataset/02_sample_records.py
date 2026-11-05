"""Sample the WLSP records of Datasets A and B.

Reads data/interim/wlsp.parquet and writes data/interim/samples.json:
{"a": [100 record IDs], "b": [300 record IDs]}. The records without any
candidate synset are dropped later (05_select_pairs.py), which leaves 86
records in Dataset A and 266 in Dataset B, of which 200 are selected.

Dataset A is drawn from the Nature division (classification numbers 1.5...),
stratified by the number of decimal digits of the classification number
(1 to 4). It was drawn in two stages: 150 records spread over the four
strata, which gave 140 because the 1-digit stratum has only 27 records, and
then 25 records from each stratum of those 140.

Dataset B is a simple random sample of 300 Nominal records.

    uv run python scripts/dataset/02_sample_records.py
"""

import json

import pandas as pd

from wlsp_bn import paths

WLSP = paths.INTERIM / "wlsp.parquet"
OUT = paths.INTERIM / "samples.json"

NATURE = "1.5"
DEPTHS = [1, 2, 3, 4]
A_FIRST_STAGE = 150
A_PER_STRATUM = 25
A_SEED = 0
B_SIZE = 300
B_SEED = 52


def decimal_digits(category_no: float) -> int:
    """The number of decimal digits of a classification number as printed."""
    text = str(category_no)
    return len(text.split(".")[1]) if "." in text else 0


def sample_a(records: pd.DataFrame) -> list[int]:
    """The 100 records of the stratified two-stage sample of Dataset A."""
    nature = records[records["category_no"].astype(str).str.startswith(NATURE)]
    nature = nature.assign(depth=nature["category_no"].map(decimal_digits))
    strata = nature[nature["depth"].isin(DEPTHS)].groupby("depth")

    # First stage: an equal share of A_FIRST_STAGE per stratum, the remainder
    # going one each to the largest strata, at most the stratum's size.
    base, extra = divmod(A_FIRST_STAGE, strata.ngroups)
    sizes = strata.size().to_dict()
    larger = set(sorted(sizes, key=sizes.get, reverse=True)[:extra])
    first = pd.concat(
        group.sample(n=min(base + (depth in larger), len(group)), random_state=A_SEED)
        for depth, group in strata
    ).sort_index()

    # Second stage: A_PER_STRATUM records from each stratum of the first.
    second = pd.concat(
        group.sample(n=A_PER_STRATUM, random_state=A_SEED)
        for _, group in first.groupby("depth")
    ).sort_index()
    return second.index.tolist()


def sample_b(records: pd.DataFrame) -> list[int]:
    """The 300 records of the simple random sample of Dataset B."""
    return records.sample(B_SIZE, random_state=B_SEED).index.tolist()


def main() -> None:
    records = pd.read_parquet(WLSP)
    samples = {"a": sample_a(records), "b": sample_b(records)}
    OUT.write_text(json.dumps(samples), encoding="utf-8")
    print(
        f"{OUT}: Dataset A {len(samples['a'])}, Dataset B {len(samples['b'])} records"
    )


if __name__ == "__main__":
    main()
