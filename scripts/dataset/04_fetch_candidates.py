"""Retrieve the candidate synsets of each record from BabelNet.

Runs in the BabelNet environment (Python 3.8), with the BabelNet RPC server
serving the local BabelNet 5.0 index; docs/dataset_construction.md says how
to set both up. Reads data/interim/queries.json and writes
data/interim/candidates.json: for each dataset, record and query type, the
sorted IDs of the synsets that a search for any of the query's terms
returns. A search is restricted to nouns in the query's language (Japanese
or English), and only concept synsets are kept, which excludes named
entities. A search that times out stops the run.

    python scripts/dataset/04_fetch_candidates.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("BABELNET_CONF", str(ROOT / "configs" / "babelnet_conf.yml"))

import babelnet as bn  # noqa: E402
from babelnet.language import Language  # noqa: E402
from babelnet.pos import POS  # noqa: E402
from babelnet.synset import SynsetType  # noqa: E402

QUERIES = ROOT / "data" / "interim" / "queries.json"
OUT = ROOT / "data" / "interim" / "candidates.json"
LANGUAGE = {
    "headword": "JA",
    "jmdict": "EN",
    "alias_ja": "JA",
    "alias_en": "EN",
    "variant_ja": "JA",
    "variant_en": "EN",
}
WORKERS = 4


def search(query: tuple[str, str]) -> list[str]:
    """The IDs of the noun concept synsets that BabelNet returns for a term."""
    term, language = query
    synsets = bn.get_synsets(
        term, from_langs=[getattr(Language, language)], poses=[POS.NOUN]
    )
    return [str(s.id) for s in synsets if s.type == SynsetType.CONCEPT]


def main() -> None:
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    searches = sorted(
        {
            (term.strip(), LANGUAGE[query_type])
            for records in queries.values()
            for terms_by_type in records.values()
            for query_type, terms in terms_by_type.items()
            for term in terms
            if term.strip()
        }
    )
    print(f"{len(searches)} searches", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        found = dict(zip(searches, pool.map(search, searches)))

    candidates: dict[str, dict[str, dict[str, list[str]]]] = {}
    for dataset, records in queries.items():
        candidates[dataset] = {
            record_id: {
                query_type: sorted(
                    {
                        sid
                        for term in terms
                        if term.strip()
                        for sid in found[(term.strip(), LANGUAGE[query_type])]
                    }
                )
                for query_type, terms in terms_by_type.items()
            }
            for record_id, terms_by_type in records.items()
        }
    OUT.write_text(json.dumps(candidates), encoding="utf-8")
    print(f"{OUT}: candidates of {sum(map(len, candidates.values()))} records")


if __name__ == "__main__":
    main()
