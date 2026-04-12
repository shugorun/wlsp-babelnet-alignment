#!/usr/bin/env python
# Usage: python src/term_expansion/term_expansion_inputs.py --wlsp data/processed/wlsp.pkl --records data/gold/gold_A_records.pkl --out-dir data/interim/api_inputs/term_expansion/version_1/gold_A
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd


DEFAULT_WLSP_PATH = Path("data/processed/wlsp.pkl")
DEFAULT_RECORDS_PATH = Path("data/gold/gold_A_records.pkl")
DEFAULT_OUT_DIR = Path("data/interim/api_inputs/term_expansion/version_1/gold_A")


def _as_list(x):
    """Convert a value into a list."""
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _clean_terms(xs):
    """Strip terms, drop empty values, and keep unique items in original order."""
    seen = set()
    out = []
    for t in _as_list(xs):
        if t is None:
            continue
        t = str(t).strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _make_category_path(row):
    """Build a category path from WLSP columns."""
    parts = []
    for col in ["category", "subcategory", "class"]:
        if col in row and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                parts.append(s)
    return ">".join(parts)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlsp", type=Path, default=DEFAULT_WLSP_PATH)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    wlsp = pd.read_pickle(args.wlsp)
    rids = pd.read_pickle(args.records).index.to_list()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_jmdict_en = 12
    max_subpara_terms = 12
    written = 0

    for rid in rids:
        row = wlsp.loc[rid]

        # Get the Japanese headword.
        lemma = (
            str(row["lemma"]).strip()
            if "lemma" in row
            else str(row.get("見出し本体", "")).strip()
        )

        kana = None
        if "kana" in row and pd.notna(row["kana"]):
            kana = str(row["kana"]).strip() or None

        category_path_ja = _make_category_path(row)
        sub_terms = _clean_terms(row.get("synonyms", []))
        jmdict_en = _clean_terms(row.get("EN_JMdict", []))

        # lemma_ja is already provided separately.
        sub_terms = [t for t in sub_terms if t != lemma]
        sub_terms = sub_terms[:max_subpara_terms]
        jmdict_en = jmdict_en[:max_jmdict_en]

        payload = {
            "record_id": rid,
            "category_path_ja": category_path_ja,
            "lemma_ja": lemma,
            "kana": kana,
            "jmdict_en": jmdict_en,
            "subparagraph_terms_ja": sub_terms,
        }

        out_path = out_dir / f"rid={rid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        written += 1

    print(f"[OK] wrote {written} files to: {out_dir}")


if __name__ == "__main__":
    main()
