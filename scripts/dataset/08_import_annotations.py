"""Read the labels from the annotated sheets.

Writes data/annotations/dataset_a.tsv and dataset_b.tsv, one row per pair
in the order of the sheets: record_id, synset_id, query_type and label.
The sheets were labelled with OK, ⊃, ⊂ and NG, which become the paper's
EQ (Equivalent), BR (Broader: the WLSP sense is broader than the synset),
NR (Narrower) and OT (Other).

Two labels of Dataset A were corrected after the annotation, with the
annotator's agreement; data/annotations/dataset_a_corrections.tsv lists them
and why. A correction whose earlier label is not the one on the sheet stops
the script.

    uv run python scripts/dataset/08_import_annotations.py \\
        --a <annotated sheet of A> --b <part 1> <part 2>
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

from wlsp_bn import paths

LABELS = {"OK": "EQ", "⊃": "BR", "⊂": "NR", "NG": "OT"}
# The sheets name the query types as the scripts that built them did.
QUERY_TYPES = {
    "sids_JA": "headword",
    "sids_JMdict": "jmdict",
    "sids_aliases_ja": "alias_ja",
    "sids_aliases_en": "alias_en",
    "sids_variants_ja": "variant_ja",
    "sids_variants_en": "variant_en",
}
CORRECTIONS = paths.ANNOTATIONS / "dataset_a_corrections.tsv"


def read_sheet(path: Path) -> pd.DataFrame:
    sheet = pd.read_excel(path)
    return pd.DataFrame(
        {
            "record_id": sheet["レコードID番号"].astype(int),
            "synset_id": sheet["BabelNet番号"].astype(str).str.strip(),
            "query_type": sheet["from"].map(lambda q: QUERY_TYPES.get(q, q)),
            "label": sheet["ラベル"].astype(str).str.strip().map(LABELS),
        }
    )


def apply_corrections(pairs: pd.DataFrame) -> pd.DataFrame:
    corrections = pd.read_csv(CORRECTIONS, sep="\t")
    for c in corrections.itertuples():
        at = (pairs["record_id"] == c.record_id) & (pairs["synset_id"] == c.synset_id)
        if at.sum() != 1 or (pairs.loc[at, "label"] != c.label_before).any():
            raise SystemExit(
                f"correction of ({c.record_id}, {c.synset_id}) does not match the sheet"
            )
        pairs.loc[at, "label"] = c.label_after
    return pairs


def write(pairs: pd.DataFrame, name: str) -> None:
    if (
        pairs["label"].isna().any()
        or pairs.duplicated(["record_id", "synset_id"]).any()
    ):
        raise SystemExit(
            f"{name}: a pair is unlabelled, has an unknown label or repeats"
        )
    path = paths.ANNOTATIONS / f"{name}.tsv"
    pairs.to_csv(
        path, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n"
    )
    n_records = pairs["record_id"].nunique()
    counts = pairs["label"].value_counts().to_dict()
    print(f"{path}: {n_records} records, {len(pairs):,} pairs, {counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--a", type=Path, required=True, help="annotated sheet of Dataset A"
    )
    ap.add_argument(
        "--b",
        type=Path,
        nargs="+",
        required=True,
        help="annotated sheets of Dataset B, in order",
    )
    args = ap.parse_args()
    paths.ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    write(apply_corrections(read_sheet(args.a)), "dataset_a")
    write(pd.concat([read_sheet(p) for p in args.b], ignore_index=True), "dataset_b")


if __name__ == "__main__":
    main()
