# src/prepare_inputs.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Iterable
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import chain

CHUNK_SIZE = 50
NA = "N/A"


def _as_list(x) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    return []


def _first_gloss(main: Any, glosses: Any) -> str:
    if isinstance(main, str) and main.strip():
        return main.strip()
    gs = _as_list(glosses)
    return (gs[0].strip() if gs and isinstance(gs[0], str) and gs[0].strip() else NA)


def _chunks(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _clean_categories_en(cats: Any) -> List[str]:
    xs = [c for c in _as_list(cats) if isinstance(c, str) and c]
    xs = [c for c in xs if not (c.startswith("All_articles_") or c.startswith("Articles_with_") or c.startswith("_"))]
    return xs if xs else [NA]


def build_candidate(babelnet: pd.DataFrame, sid: str) -> Dict[str, Any] | None:
    if sid not in babelnet.index:
        return None

    r = babelnet.loc[sid]
    lemmas_ja = _as_list(r.get("lemmas_JA"))[:3]
    lemmas_en = _as_list(r.get("lemmas_EN"))[:3]
    if not lemmas_ja and not lemmas_en:
        return None

    hyper_lists = [
        (_as_list(babelnet.at[hsid, "lemmas_EN"])[:3] if hsid in babelnet.index else [])
        for hsid in _as_list(r.get("hypernym_sids"))
    ]
    hypernym_lemmas_en = list(dict.fromkeys(chain.from_iterable(hyper_lists))) or [NA]

    return {
        "synset_id": sid,
        "lemmas_ja": lemmas_ja,
        "lemmas_en": lemmas_en,
        "gloss_ja": _first_gloss(r.get("main_gloss_JA"), r.get("glosses_JA")),
        "gloss_en": _first_gloss(r.get("main_gloss_EN"), r.get("glosses_EN")),
        "categories_en": _clean_categories_en(r.get("categories_EN")),
        "hypernym_lemmas_en": hypernym_lemmas_en,
    }


def build_record(records: pd.DataFrame, babelnet: pd.DataFrame, rid: int) -> Dict[str, Any]:
    row = records.loc[rid]
    sids_ja = _as_list(row.get("sids_JA"))
    sids_jmd = _as_list(row.get("sids_JMdict"))
    sids = list(dict.fromkeys(sids_ja + [sid for sid in sids_jmd if sid not in set(sids_ja)]))

    candidates = [c for sid in sids if (c := build_candidate(babelnet, sid)) is not None]

    return {
        "record_id": str(rid),
        "wlsp": {
            "headword_ja": row.get("lemma", ""),
            "synonyms_ja": _as_list(row.get("synonyms")),
            "category_path_ja": " > ".join(
                [x for x in [row.get("category"), row.get("subcategory"), row.get("class")] if isinstance(x, str) and x]
            ),
            "category_no": str(row.get("category_no", "")),
        },
        "candidates": candidates,
    }


def main():
    # setting directory for outputs
    OUT_DIR = "data/interim/api_inputs/alignment/version_1/gold_A-ex"
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load rids
    gold = pd.read_pickle("data/gold/gold_A_records.pkl")
    rids = gold.index.tolist()

    # load babelnet
    babelnet = pd.read_pickle("data/processed/babelnet_.pkl")

    # build inputs
    manifest = []
    for rid in rids:
        rec = build_record(gold, babelnet, rid)
        cands = rec["candidates"]

        k = 0
        for k, chunk in enumerate(_chunks(cands, CHUNK_SIZE), start=1):
            payload = {
                "record_id": rec["record_id"],
                "wlsp": rec["wlsp"],
                "chunk": {"index": k - 1, "size": len(chunk), "total_candidates": len(cands), "chunk_size": CHUNK_SIZE},
                "candidates": chunk,
            }
            (out_dir / f"rid={rec['record_id']}_chunk={k-1:03d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        manifest.append({"record_id": rec["record_id"], "total_candidates": len(cands), "num_chunks": (k if cands else 0)})

    pd.DataFrame(manifest).to_csv(f"{OUT_DIR}/manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {len(manifest)} records to: {OUT_DIR}")
    print(f"[OK] manifest: {OUT_DIR}/manifest.csv")


if __name__ == "__main__":
    main()
