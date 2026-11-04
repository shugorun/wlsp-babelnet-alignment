"""Turn the candidates into record-synset pairs and select the records of Dataset B.

Reads data/interim/samples.json and data/interim/candidates.json and writes
data/interim/pairs.json: for each dataset, the list of
[record_id, synset_id, query_type] in the order of the annotation sheets.
A synset retrieved by several query types appears once, under the first
type in QUERY_TYPES. Records without any candidate are dropped: this leaves
86 of the 100 sampled records in Dataset A and 266 of the 300 in Dataset B,
of which 200 are selected at random.

Within a query type the synsets are in the order of their IDs.

    uv run python scripts/dataset/05_select_pairs.py
"""

import json
import random

from wlsp_bn import paths

SAMPLES = paths.INTERIM / "samples.json"
CANDIDATES = paths.INTERIM / "candidates.json"
OUT = paths.INTERIM / "pairs.json"

QUERY_TYPES = ["headword", "jmdict", "alias_ja", "alias_en", "variant_ja", "variant_en"]
B_SELECTED = 200
B_SEED = 242


def record_pairs(record_id: int, candidates: dict[str, list[str]]) -> list[list]:
    """The pairs of one record, each synset under the first query type that found it."""
    seen = set()
    pairs = []
    for query_type in QUERY_TYPES:
        for sid in candidates.get(query_type, []):
            if sid not in seen:
                seen.add(sid)
                pairs.append([record_id, sid, query_type])
    return pairs


def main() -> None:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))

    by_record = {
        dataset: {
            i: p for i in ids if (p := record_pairs(i, candidates[dataset][str(i)]))
        }
        for dataset, ids in samples.items()
    }
    found_b = list(by_record["b"])
    print(f"Dataset B: {len(found_b)} of {len(samples['b'])} records have candidates")
    chosen = set(random.Random(B_SEED).sample(found_b, B_SELECTED))
    by_record["b"] = {i: p for i, p in by_record["b"].items() if i in chosen}

    pairs = {
        dataset: [p for ps in recs.values() for p in ps]
        for dataset, recs in by_record.items()
    }
    OUT.write_text(json.dumps(pairs), encoding="utf-8")
    for dataset, recs in by_record.items():
        n_pairs = len(pairs[dataset])
        print(f"Dataset {dataset.upper()}: {len(recs)} records, {n_pairs:,} pairs")


if __name__ == "__main__":
    main()
