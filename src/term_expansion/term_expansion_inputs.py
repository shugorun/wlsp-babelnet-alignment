from pathlib import Path
import json

import numpy as np
import pandas as pd


# --- load data ---
# Read WLSP table from pickle.
wlsp = pd.read_pickle("data/processed/wlsp.pkl")

# Read candidate records from pickle and get their record IDs from the index.
rids = pd.read_pickle("data/gold/gold_A_records.pkl").index.to_list()


# --- output dir ---
# Create the output directory if it does not exist.
out_dir = Path("data/interim/api_inputs/term_expansion/version_1/gold_A")
out_dir.mkdir(parents=True, exist_ok=True)


# --- helpers ---
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


# --- build & save ---
# Limit the number of hint terms to avoid oversized inputs.
MAX_JMDICT_EN = 12
MAX_SUBPARA_TERMS = 12

written = 0

for rid in rids:
    row = wlsp.loc[rid]

    # Get the Japanese headword.
    lemma = (
        str(row["lemma"]).strip()
        if "lemma" in row
        else str(row.get("見出し本体", "")).strip()
    )

    # Get kana if available.
    kana = None
    if "kana" in row and pd.notna(row["kana"]):
        kana = str(row["kana"]).strip() or None

    # Build the Japanese category path.
    category_path_ja = _make_category_path(row)

    # Collect hint terms.
    sub_terms = _clean_terms(row.get("synonyms", []))
    jmdict_en = _clean_terms(row.get("EN_JMdict", []))

    # Remove the lemma itself from subparagraph_terms_ja,
    # because lemma_ja is already given separately.
    sub_terms = [t for t in sub_terms if t != lemma]

    # Truncate long lists while keeping original order.
    sub_terms = sub_terms[:MAX_SUBPARA_TERMS]
    jmdict_en = jmdict_en[:MAX_JMDICT_EN]

    # Build the JSON payload.
    payload = {
        "record_id": rid,
        "category_path_ja": category_path_ja,
        "lemma_ja": lemma,
        "kana": kana,
        "jmdict_en": jmdict_en,
        "subparagraph_terms_ja": sub_terms,
    }

    # Save one JSON file per record.
    out_path = out_dir / f"rid={rid}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    written += 1

print(f"[OK] wrote {written} files to: {out_dir}")