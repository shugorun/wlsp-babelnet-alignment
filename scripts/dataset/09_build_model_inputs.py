"""Build the text of each pair that the models are given.

Reads data/annotations/dataset_{a,b}.tsv, data/interim/wlsp.parquet and
data/interim/synsets.json and writes data/interim/pairs_{a,b}.parquet, one
row per pair in the order of the annotations: record_id, synset_id,
query_type, label, and the record and synset text at two levels of detail:

    wlsp_min   the headword and its reading
    wlsp_full  the WLSP cell of the sheets
    bn_1step   the synset's lemmas, Japanese categories and glosses
    bn_full    those, then its relations, its IS A targets two steps up and
               its Chinese lemmas, categories and glosses

Both datasets get the text of the sheets of Dataset B, also for Dataset A,
whose annotator saw the shorter description of src/wlsp_bn/cells.py's
synset_text_a.

    uv run python scripts/dataset/09_build_model_inputs.py
"""

import json

import pandas as pd

from wlsp_bn import paths
from wlsp_bn.cells import headword_text, joined_synset_cells, record_text, synset_cells

WLSP = paths.INTERIM / "wlsp.parquet"
SYNSETS = paths.INTERIM / "synsets.json"


def main() -> None:
    records = pd.read_parquet(WLSP)
    synsets = json.loads(SYNSETS.read_text(encoding="utf-8"))
    for dataset in ["a", "b"]:
        pairs = pd.read_csv(paths.ANNOTATIONS / f"dataset_{dataset}.tsv", sep="\t")
        rows = []
        for p in pairs.itertuples():
            record = records.loc[p.record_id]
            cells = synset_cells(synsets, p.synset_id)
            rows.append(
                {
                    "record_id": p.record_id,
                    "synset_id": p.synset_id,
                    "query_type": p.query_type,
                    "label": p.label,
                    "wlsp_min": headword_text(record),
                    "wlsp_full": record_text(record),
                    "bn_1step": cells["BabelNet情報"],
                    "bn_full": joined_synset_cells(cells),
                }
            )
        out = paths.INTERIM / f"pairs_{dataset}.parquet"
        pd.DataFrame(rows).to_parquet(out)
        print(f"{out}: {len(rows):,} pairs")


if __name__ == "__main__":
    main()
