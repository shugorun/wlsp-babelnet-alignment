#!/usr/bin/env python
# Usage: python src/babelnet_pipeline/add_sids_to_wlsp.py --input-path data/gold/gold_A_records.parquet --outputs-dir outputs/api_runs/term_expansion/version_1/gold_A
# add_sids.py
# Add sid columns to a candidate table and overwrite the input parquet file.

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import zerorpc  # type: ignore

try:
    import babelnet as bn  # type: ignore
    from babelnet.language import Language  # type: ignore
    from babelnet.pos import POS  # type: ignore
    from babelnet.synset import SynsetType  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import BabelNet modules. Make sure babelnet is available."
    ) from e

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = ROOT / "data" / "gold" / "gold_A_records.parquet"
DEFAULT_OUTPUTS_DIR = ROOT / "outputs" / "api_runs" / "term_expansion" / "version_1" / "gold_A"
MAX_WORKERS = 4

# source term column -> language -> output sid column
FIELD_SPECS = [
    ("lemma", "JA", "sids_JA"),
    ("EN_JMdict", "EN", "sids_JMdict"),
    ("aliases_ja", "JA", "sids_aliases_ja"),
    ("aliases_en", "EN", "sids_aliases_en"),
    ("variants_ja", "JA", "sids_variants_ja"),
    ("variants_en", "EN", "sids_variants_en"),
    ("hypernyms_ja", "JA", "sids_hypernyms_ja"),
    ("hypernyms_en", "EN", "sids_hypernyms_en"),
]

BASE_COLUMNS = ["lemma", "EN_JMdict"]
EXPANDED_TERM_COLUMNS = [
    "aliases_ja",
    "aliases_en",
    "variants_ja",
    "variants_en",
    "hypernyms_ja",
    "hypernyms_en",
]


def fetch_sids_ja(word: str) -> list[str]:
    """Fetch Japanese noun concept synsets."""
    sids = []
    try:
        synsets = bn.get_synsets(word, from_langs=[Language.JA], poses=[POS.NOUN])
        for synset in synsets:
            if synset.type == SynsetType.CONCEPT:
                sid = synset.id
                sids.append(sid if isinstance(sid, str) else str(sid))
    except zerorpc.exceptions.TimeoutExpired:
        print(f"TIMEOUT (JA): {word}")
    return sids


def fetch_sids_en(word: str) -> list[str]:
    """Fetch English noun concept synsets."""
    sids = []
    try:
        synsets = bn.get_synsets(word, from_langs=[Language.EN], poses=[POS.NOUN])
        for synset in synsets:
            if synset.type == SynsetType.CONCEPT:
                sid = synset.id
                sids.append(sid if isinstance(sid, str) else str(sid))
    except zerorpc.exceptions.TimeoutExpired:
        print(f"TIMEOUT (EN): {word}")
    return sids


def is_missing(x: object) -> bool:
    """Return True only for scalar NA-like values."""
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        return False
    return bool(pd.isna(x))


def normalize_terms(value: object) -> list[str]:
    """Convert one cell into a clean list of terms."""
    if value is None or is_missing(value):
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        out = []
        for item in list(value):
            if item is None or is_missing(item):
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    text = str(value).strip()
    return [text] if text else []


def parallel_map(func, items: list[str], max_workers: int) -> list[list[str]]:
    """Run lookup in parallel and keep input order."""
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        return list(tqdm(ex.map(func, items), total=len(items), desc="lookup", unit="term"))


def build_sid_series(series_in: pd.Series, lang: str, out_name: str, max_workers: int) -> pd.Series:
    """Build one sid list per record from one source term column."""
    term_lists = series_in.map(normalize_terms).tolist()

    flat_terms = []
    lengths = []
    for terms in term_lists:
        flat_terms.extend(terms)
        lengths.append(len(terms))

    if not flat_terms:
        return pd.Series([[] for _ in range(len(series_in))], index=series_in.index, name=out_name)

    fetcher = fetch_sids_en if lang == "EN" else fetch_sids_ja
    flat_sids = parallel_map(fetcher, flat_terms, max_workers=max_workers)

    merged_sids = []
    start = 0
    for length in lengths:
        end = start + length
        row_sids = []
        for sid_list in flat_sids[start:end]:
            row_sids.extend(sid_list)
        merged_sids.append(sorted(set(row_sids)))
        start = end

    return pd.Series(merged_sids, index=series_in.index, name=out_name)


def parsed_record_path(outputs_dir: Path, rid: str) -> Path:
    """Return the parsed JSON path for one record."""
    return outputs_dir / "parsed" / "records" / f"rid={rid}.json"


def load_missing_term_columns(
    df: pd.DataFrame,
    outputs_dir: Path,
    term_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Temporarily restore missing expanded term columns from parsed JSON.

    Returned columns are marked so they can be dropped before saving.
    """
    missing_columns = [col for col in term_columns if col not in df.columns]
    if not missing_columns:
        return df, []

    print("Restoring missing term columns: " + ", ".join(missing_columns))

    values_by_col = {col: [] for col in missing_columns}
    missing_files = 0

    for rid in tqdm(df.index.tolist(), desc="loading parsed outputs", unit="record"):
        path = parsed_record_path(outputs_dir, str(rid))
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
        else:
            parsed = {}
            missing_files += 1

        for col in missing_columns:
            values_by_col[col].append(normalize_terms(parsed.get(col, [])))

    if missing_files == len(df):
        raise FileNotFoundError(
            f"No parsed JSON files were found under {outputs_dir / 'parsed' / 'records'}"
        )

    if missing_files > 0:
        print(f"WARNING: parsed JSON was missing for {missing_files} records.")

    for col, values in values_by_col.items():
        df[col] = pd.Series(values, index=df.index, name=col)

    return df, missing_columns


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Load the input table. The index is assumed to be record_id.
    df = pd.read_parquet(args.input_path)
    print(f"Loaded {len(df)} records from {args.input_path}")

    # Base columns must already exist.
    missing_base_columns = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing_base_columns:
        raise KeyError("Missing required columns: " + ", ".join(missing_base_columns))

    # Missing expanded term columns are loaded only for sid creation.
    df, temporary_columns = load_missing_term_columns(
        df,
        outputs_dir=args.outputs_dir,
        term_columns=EXPANDED_TERM_COLUMNS,
    )

    # Rebuild each sid column.
    for src_col, lang, out_col in FIELD_SPECS:
        print(f"Building {out_col} from {src_col} ({lang})...")
        df[out_col] = build_sid_series(
            df[src_col],
            lang=lang,
            out_name=out_col,
            max_workers=args.max_workers,
        )

    # Do not keep temporarily restored term columns in the saved file.
    if temporary_columns:
        df = df.drop(columns=temporary_columns)

    # Overwrite the original parquet file.
    df.to_parquet(args.input_path)
    print(f"Updated and saved: {args.input_path}")


if __name__ == "__main__":
    main()
