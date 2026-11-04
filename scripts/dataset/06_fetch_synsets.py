"""Fetch from BabelNet what the sheets and model inputs show of each candidate synset.

Runs in the BabelNet environment (Python 3.8), with the BabelNet RPC server
serving the local BabelNet 5.0 index; docs/dataset_construction.md says how
to set both up. Reads data/interim/pairs.json and writes
data/interim/synsets.json in the form that src/wlsp_bn/cells.py reads:

- for each candidate synset: its lemmas, categories and glosses in
  Japanese, English and Chinese, its Chinese main gloss, the targets of the
  relations in RELATIONS, and its hypernyms as the sheet of Dataset A
  showed them (the targets of the first of HYPERNYMS_A that has any);
- for each target of those: its Japanese and English lemmas, and for the
  targets of IS A also their own IS A targets;
- for each of those second IS A targets: its Japanese and English lemmas.

Each stage fetches a synset once and appends its results to
data/interim/synsets.<stage>.jsonl as they come. A request that times out
stops the run, and running the script again continues where it stopped. A
part of a synset on which the RPC server returns an error, as it does for
the glosses of some synsets, is left empty.

    python scripts/dataset/06_fetch_synsets.py
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("BABELNET_CONF", str(ROOT / "configs" / "babelnet_conf.yml"))

import babelnet as bn  # noqa: E402
import zerorpc  # noqa: E402
from babelnet.data.relation import BabelPointer  # noqa: E402
from babelnet.language import Language  # noqa: E402
from babelnet.resources import BabelSynsetID  # noqa: E402

INTERIM = ROOT / "data" / "interim"
PAIRS = INTERIM / "pairs.json"
OUT = INTERIM / "synsets.json"
WORKERS = 4

# Each relation shown on the sheets of Dataset B, the BabelNet pointers it
# collects (targets in this order, without repeats) and the most targets kept.
RELATIONS = [
    ("ANTONIM", ["ANTONYM"], None),
    (
        "IS A",
        [
            "ANY_HYPERNYM",
            "HYPERNYM",
            "HYPERNYM_INSTANCE",
            "WIBI_HYPERNYM",
            "WIKIDATA_HYPERNYM",
        ],
        None,
    ),
    (
        "PART OF",
        [
            "ANY_MERONYM",
            "MERONYM_MEMBER",
            "MERONYM_PART",
            "MERONYM_SUBSTANCE",
            "WIKIDATA_MERONYM",
        ],
        None,
    ),
    (
        "HAS KIND",
        [
            "ANY_HYPONYM",
            "WIKIDATA_HYPONYM_INSTANCE",
            "WIKIDATA_HYPONYM",
            "WIBI_HYPONYM",
        ],
        3,
    ),
    ("HAS PART", ["HOLONYM_PART"], None),
    ("HAS INSTANCE", ["HYPONYM_INSTANCE"], None),
    ("HAS MEMBER", ["HOLONYM_MEMBER"], None),
    ("SUBSTANCE", ["HOLONYM_SUBSTANCE"], None),
    ("CAUSES", ["CAUSE"], None),
    ("DIFFERENT FROM", ["SIMILAR_TO"], None),
    ("VERB GROUP", ["VERB_GROUP"], None),
    ("USAGE", ["USAGE"], None),
    ("REGION", ["REGION"], None),
]
IS_A_POINTERS = RELATIONS[1][1]
# The sheet of Dataset A showed the targets of the first of these pointers
# that has any, as they come.
HYPERNYMS_A = [
    "HYPERNYM",
    "HYPERNYM_INSTANCE",
    "WIKIDATA_HYPERNYM",
    "WIBI_HYPERNYM",
    "ANY_HYPERNYM",
]
LANGUAGES = {"JA": Language.JA, "EN": Language.EN, "ZH": Language.ZH}


def get_synset(sid: str):
    """The synset, or None when the RPC server returns an error on it."""
    try:
        return bn.get_synset(BabelSynsetID(sid))
    except zerorpc.exceptions.RemoteError as e:
        print(f"get_synset failed: {sid} {e!r}", flush=True)
        return None


def logged(what: str, sid: str, call: Callable, default):
    """The result of `call`, or `default` when the RPC server returns an error."""
    try:
        return call()
    except zerorpc.exceptions.RemoteError as e:
        print(f"{what} failed: {sid} {e!r}", flush=True)
        return default


def pointer(name: str) -> BabelPointer | None:
    if name in BabelPointer.__members__:
        return BabelPointer[name]
    matches = BabelPointer.from_name(name)
    return matches[0] if matches else None


def targets(synset, pointer_names: list[str], limit: int | None = None) -> list[str]:
    """The targets of the pointers, in order and without repeats, at most `limit`."""
    found: list[str] = []
    for name in pointer_names:
        p = pointer(name)
        if p is None:
            print(f"unknown pointer: {name}", flush=True)
            continue
        for edge in synset.outgoing_edges(p):
            target = str(edge.id_target)
            if target not in found:
                found.append(target)
                if limit is not None and len(found) >= limit:
                    return found
    return found


def hypernyms_a(synset) -> list[str]:
    for name in HYPERNYMS_A:
        found = [
            str(edge.id_target) for edge in synset.outgoing_edges(BabelPointer[name])
        ]
        if found:
            return found
    return []


def lemmas(synset, languages: list[str]) -> dict[str, list[str]]:
    return {
        lang: [str(x.lemma) for x in synset.lemmas(LANGUAGES[lang])]
        for lang in languages
    }


def candidate_entry(sid: str) -> dict | None:
    synset = get_synset(sid)
    if synset is None:
        return None
    zh = logged(
        "main gloss", sid, lambda: synset.main_gloss(language=Language.ZH), None
    )
    return {
        "lemmas": lemmas(synset, ["JA", "EN", "ZH"]),
        "categories": {
            lang: [str(c.category) for c in synset.categories(code)]
            for lang, code in LANGUAGES.items()
        },
        "glosses": {
            lang: logged(
                "glosses",
                sid,
                lambda code=code: [str(g.gloss) for g in synset.glosses(language=code)],
                [],
            )
            for lang, code in LANGUAGES.items()
        },
        "main_gloss": {"ZH": None if zh is None else str(zh.gloss)},
        "relations": {
            label: logged(
                label, sid, lambda n=names, m=limit: targets(synset, n, m), []
            )
            for label, names, limit in RELATIONS
        },
        "hypernyms": logged("hypernyms", sid, lambda: hypernyms_a(synset), []),
    }


def target_entry(sid: str) -> dict | None:
    synset = get_synset(sid)
    return None if synset is None else {"lemmas": lemmas(synset, ["JA", "EN"])}


def is_a_target_entry(sid: str) -> dict | None:
    synset = get_synset(sid)
    if synset is None:
        return None
    return {
        "lemmas": lemmas(synset, ["JA", "EN"]),
        "relations": {
            "IS A": logged("IS A", sid, lambda: targets(synset, IS_A_POINTERS), [])
        },
    }


def fetch(stage: str, fetch_one: Callable[[str], dict | None], sids: list[str]) -> dict:
    """Fetch the entries of `sids`, reusing those a stopped run already saved."""
    progress = INTERIM / f"synsets.{stage}.jsonl"
    done: dict[str, dict | None] = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["synset_id"]] = row["entry"]
    todo = [s for s in sids if s not in done]
    print(f"{stage}: {len(sids)} synsets, {len(todo)} to fetch", flush=True)
    pool = ProcessPoolExecutor(WORKERS)
    with progress.open("a", encoding="utf-8") as f, pool:
        for sid, entry in zip(todo, pool.map(fetch_one, todo)):
            row = {"synset_id": sid, "entry": entry}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[sid] = entry
    return done


def main() -> None:
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    candidates = sorted({sid for rows in pairs.values() for _, sid, _ in rows})
    synsets = {
        s: e for s, e in fetch("candidates", candidate_entry, candidates).items() if e
    }

    is_a_first = sorted({t for e in synsets.values() for t in e["relations"]["IS A"]})
    other_first = sorted(
        {t for e in synsets.values() for ts in e["relations"].values() for t in ts}
        | {t for e in synsets.values() for t in e["hypernyms"]}
    )
    first = fetch("is_a_targets", is_a_target_entry, is_a_first)
    first.update(
        fetch("targets", target_entry, [s for s in other_first if s not in first])
    )
    second = sorted(
        {t for e in first.values() if e for t in e.get("relations", {}).get("IS A", [])}
    )
    seen = set(synsets) | set(first)
    first.update(
        fetch("is_a_targets_2", target_entry, [s for s in second if s not in seen])
    )

    for sid, entry in first.items():
        if entry and sid not in synsets:
            synsets[sid] = entry
    OUT.write_text(json.dumps(synsets, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT}: {len(candidates)} candidate synsets, {len(synsets)} synsets in all")


if __name__ == "__main__":
    main()
