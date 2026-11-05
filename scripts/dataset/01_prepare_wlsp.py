"""Read the Nominal records of WLSP and add what the later steps need.

Reads the WLSP database and JMdict (configs/paths.toml: wlsp_csv,
jmdict_xml) and writes data/interim/wlsp.parquet: one row per Nominal record
(66,884 records), with its co-classified words and the English terms that
JMdict gives for its headword.

    uv run python scripts/dataset/01_prepare_wlsp.py
"""

from wlsp_bn import paths
from wlsp_bn.wlsp import co_classified_words, jmdict_index, read_nominal_records

OUT = paths.INTERIM / "wlsp.parquet"


def main() -> None:
    records = read_nominal_records(paths.resource("wlsp_csv"))
    records["co_classified"] = co_classified_words(records)
    index = jmdict_index(paths.resource("jmdict_xml"))
    records["jmdict_en"] = [index.get(w, []) for w in records["headword"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(OUT)
    print(f"{OUT}: {len(records):,} Nominal records")


if __name__ == "__main__":
    main()
