# add_babelnet_rows.py
# Read sid columns from the candidate table, fetch missing BabelNet rows,
# and append them to the BabelNet table.

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import zerorpc  # type: ignore

try:
    import babelnet as bn  # type: ignore
    from babelnet.language import Language  # type: ignore
    from babelnet.resources import BabelSynsetID  # type: ignore
    from babelnet.data.relation import BabelPointer  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import BabelNet modules. Make sure babelnet is available."
    ) from e


INPUT_PATH = Path("data/gold/gold_A_records.parquet")
BABELNET_PATH = Path("data/processed/babelnet_.parquet")
MAX_WORKERS = 4
INCLUDE_HYPERNYMS = True

HYPERNYM_POINTER_NAMES = [
    "HYPERNYM",
    "HYPERNYM_INSTANCE",
    "WIKIDATA_HYPERNYM",
    "WIBI_HYPERNYM",
    "ANY_HYPERNYM",
]


def parallel_map(func, items: list[str], max_workers: int) -> list:
    """Run a function in parallel and keep input order."""
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        return list(tqdm(ex.map(func, items), total=len(items), desc="processing", unit="task"))


def is_missing(x: object) -> bool:
    """Return True only for scalar NA-like values."""
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        return False
    return bool(pd.isna(x))


def normalize_sid_container(value: object) -> list[str]:
    """Convert one sid cell into a clean sid list."""
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
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text:
                out.append(text)
        return out

    text = str(value).strip()
    return [text] if text else []


def dedupe_sorted(strings) -> list[str]:
    """Return unique strings in sorted order."""
    return sorted(set(strings))


def detect_sid_columns(df: pd.DataFrame) -> list[str]:
    """Detect sid columns such as sids_JA and sids_JMdict."""
    return [col for col in df.columns if isinstance(col, str) and col.startswith("sids_")]


def collect_sids_from_columns(df: pd.DataFrame, sid_columns: list[str]) -> list[str]:
    """Collect all unique sid values from the detected sid columns."""
    all_sids = []
    for col in sid_columns:
        for value in df[col].tolist():
            all_sids.extend(normalize_sid_container(value))
    return dedupe_sorted(all_sids)


def fetch_hypernym_sids(sid: str) -> list[str]:
    """Fetch direct hypernym sid list for one synset."""
    hypernym_sids = []
    try:
        synset = bn.get_synset(BabelSynsetID(sid))
        for pointer_name in HYPERNYM_POINTER_NAMES:
            for edge in synset.outgoing_edges(BabelPointer[pointer_name]):
                target_sid = edge.id_target
                hypernym_sids.append(target_sid if isinstance(target_sid, str) else str(target_sid))
    except zerorpc.exceptions.TimeoutExpired:
        print(f"TIMEOUT fetch_hypernym_sids: {sid}")
        return []
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR fetch_hypernym_sids: {sid} {e!r}")
        return []
    except Exception as e:
        print(f"ERROR fetch_hypernym_sids: {sid} {e!r}")
        return []

    return dedupe_sorted(hypernym_sids)


def fetch_synset_info(sid: str) -> dict | None:
    """Fetch BabelNet metadata for one synset."""
    categories_JA = []
    categories_EN = []
    lemmas_JA = []
    lemmas_EN = []
    glosses_JA = []
    glosses_EN = []
    main_gloss_JA = None
    main_gloss_EN = None

    try:
        synset = bn.get_synset(BabelSynsetID(sid))
    except zerorpc.exceptions.TimeoutExpired:
        print(f"TIMEOUT fetch_synset_info: {sid}")
        return None
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR get_synset: {sid} {e!r}")
        return None
    except Exception as e:
        print(f"ERROR get_synset: {sid} {e!r}")
        return None

    try:
        for category in synset.categories(Language.JA):
            categories_JA.append(category.category if isinstance(category.category, str) else str(category.category))
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR categories JA: {sid} {e!r}")

    try:
        for category in synset.categories(Language.EN):
            categories_EN.append(category.category if isinstance(category.category, str) else str(category.category))
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR categories EN: {sid} {e!r}")

    try:
        for lemma in synset.lemmas(Language.JA):
            lemmas_JA.append(lemma.lemma if isinstance(lemma.lemma, str) else str(lemma.lemma))
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR lemmas JA: {sid} {e!r}")

    try:
        for lemma in synset.lemmas(Language.EN):
            lemmas_EN.append(lemma.lemma if isinstance(lemma.lemma, str) else str(lemma.lemma))
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR lemmas EN: {sid} {e!r}")

    try:
        gloss = synset.main_gloss(language=Language.JA)
        if gloss is not None:
            main_gloss_JA = gloss.gloss if isinstance(gloss.gloss, str) else str(gloss.gloss)
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR main_gloss JA: {sid} {e!r}")

    try:
        gloss = synset.main_gloss(language=Language.EN)
        if gloss is not None:
            main_gloss_EN = gloss.gloss if isinstance(gloss.gloss, str) else str(gloss.gloss)
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR main_gloss EN: {sid} {e!r}")

    try:
        for gloss in synset.glosses(language=Language.JA):
            glosses_JA.append(gloss.gloss if isinstance(gloss.gloss, str) else str(gloss.gloss))
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR glosses JA: {sid} {e!r}")

    try:
        for gloss in synset.glosses(language=Language.EN):
            glosses_EN.append(gloss.gloss if isinstance(gloss.gloss, str) else str(gloss.gloss))
    except zerorpc.exceptions.RemoteError as e:
        print(f"REMOTE ERROR glosses EN: {sid} {e!r}")

    return {
        "categories_JA": dedupe_sorted(categories_JA),
        "categories_EN": dedupe_sorted(categories_EN),
        "lemmas_JA": dedupe_sorted(lemmas_JA),
        "lemmas_EN": dedupe_sorted(lemmas_EN),
        "glosses_JA": dedupe_sorted(glosses_JA),
        "glosses_EN": dedupe_sorted(glosses_EN),
        "main_gloss_JA": main_gloss_JA,
        "main_gloss_EN": main_gloss_EN,
    }


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the canonical BabelNet columns exist."""
    required_columns = {
        "hypernym_sids": [],
        "categories_JA": [],
        "categories_EN": [],
        "lemmas_JA": [],
        "lemmas_EN": [],
        "glosses_JA": [],
        "glosses_EN": [],
        "main_gloss_JA": None,
        "main_gloss_EN": None,
    }

    for col, default in required_columns.items():
        if col not in df.columns:
            if isinstance(default, list):
                df[col] = pd.Series([[] for _ in range(len(df))], index=df.index, name=col)
            else:
                df[col] = pd.Series([default for _ in range(len(df))], index=df.index, name=col)

    return df


def append_missing_babelnet_rows() -> None:
    """Read the candidate table, fetch missing BabelNet rows, and save the updated table."""
    cands = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(cands)} records from {INPUT_PATH}")

    sid_columns = detect_sid_columns(cands)
    if not sid_columns:
        raise KeyError("No sid columns were found. Expected columns starting with 'sids_'.")

    print("Detected sid columns:", ", ".join(sid_columns))

    all_candidate_sids = collect_sids_from_columns(cands, sid_columns)
    print(f"Collected {len(all_candidate_sids)} unique sids from the candidate table")

    babelnet = pd.read_parquet(BABELNET_PATH)
    babelnet.index = babelnet.index.map(str)
    babelnet.index.name = "synset_id"
    babelnet = ensure_required_columns(babelnet)
    print(f"Loaded {len(babelnet)} existing BabelNet rows from {BABELNET_PATH}")

    missing_direct_sids = sorted(set(all_candidate_sids) - set(babelnet.index))
    print(f"Missing direct sids: {len(missing_direct_sids)}")

    target_sids = set(missing_direct_sids)
    direct_hypernyms_by_sid = {}

    if INCLUDE_HYPERNYMS and missing_direct_sids:
        print("Fetching hypernym sids for missing direct sids...")
        hypernym_lists = parallel_map(fetch_hypernym_sids, missing_direct_sids, max_workers=MAX_WORKERS)
        direct_hypernyms_by_sid = {
            sid: dedupe_sorted(hypernym_sids)
            for sid, hypernym_sids in zip(missing_direct_sids, hypernym_lists)
        }

        hypernym_targets = {
            hyper_sid
            for hypernym_sids in direct_hypernyms_by_sid.values()
            for hyper_sid in hypernym_sids
        }
        missing_hypernym_sids = sorted(hypernym_targets - set(babelnet.index) - target_sids)
        target_sids.update(missing_hypernym_sids)
        print(f"Additional missing hypernym target sids: {len(missing_hypernym_sids)}")

    target_sid_list = sorted(target_sids)
    if not target_sid_list:
        print("No new BabelNet rows were needed.")
        return

    print(f"Fetching synset info for {len(target_sid_list)} new sids...")
    info_list = parallel_map(fetch_synset_info, target_sid_list, max_workers=MAX_WORKERS)

    rows = {}
    failed_sids = []
    for sid, info in zip(target_sid_list, info_list):
        if info is None:
            failed_sids.append(sid)
            continue
        rows[sid] = info

    if not rows:
        raise RuntimeError("Failed to fetch info for every missing sid.")

    additional_babelnet = pd.DataFrame.from_dict(rows, orient="index")
    additional_babelnet.index.name = "synset_id"
    additional_babelnet = ensure_required_columns(additional_babelnet)

    hypernym_series = []
    for sid in additional_babelnet.index.tolist():
        if sid in direct_hypernyms_by_sid:
            hypernym_series.append(direct_hypernyms_by_sid[sid])
        else:
            hypernym_series.append(fetch_hypernym_sids(sid))

    additional_babelnet["hypernym_sids"] = pd.Series(
        hypernym_series,
        index=additional_babelnet.index,
        name="hypernym_sids",
    )

    overlapping = sorted(set(additional_babelnet.index) & set(babelnet.index))
    if overlapping:
        print(f"WARNING: {len(overlapping)} rows already existed and will be skipped.")
        additional_babelnet = additional_babelnet.loc[~additional_babelnet.index.isin(overlapping)]

    if additional_babelnet.empty:
        print("No appendable rows remained after filtering overlaps.")
        return

    babelnet = pd.concat([babelnet, additional_babelnet], axis=0).sort_index()
    babelnet.index.name = "synset_id"

    print(f"Appended {len(additional_babelnet)} rows")
    print(f"Updated BabelNet size: {len(babelnet)}")
    if failed_sids:
        print(f"WARNING: failed to fetch {len(failed_sids)} sids")

    babelnet.to_parquet(BABELNET_PATH)
    print(f"Saved: {BABELNET_PATH}")


def main() -> None:
    append_missing_babelnet_rows()


if __name__ == "__main__":
    main()