"""The text that describes a WLSP record or a BabelNet synset.

The same text fills the cells of the annotation sheets and is given to the
models. The functions are pure: the BabelNet side reads a synset table
made by scripts/dataset/06_fetch_synsets.py, which maps a synset ID to

    {"lemmas":     {"JA": [...], "EN": [...], "ZH": [...]},
     "categories": {"JA": [...], "EN": [...], "ZH": [...]},
     "glosses":    {"JA": [...], "EN": [...], "ZH": [...]},
     "main_gloss": {"ZH": str | None},
     "relations":  {label: [target synset IDs]},
     "hypernyms":  [synset IDs]}

with only the keys that were fetched for that synset: relation targets,
for example, have only their lemmas, and only the candidate synsets have
relations. A synset that BabelNet did not return has no entry.

Two formats exist, because the two datasets were annotated at different
times. The format of Dataset B (synset_cells) was also used to build the
model inputs of both datasets; the earlier format (synset_text_a) appears
only on the annotation sheet of Dataset A.
"""

from collections.abc import Iterable, Mapping

import pandas as pd

Synsets = Mapping[str, Mapping]

# The relations shown on the sheets of Dataset B, in the order shown.
RELATIONS = [
    "ANTONIM",
    "IS A",
    "PART OF",
    "HAS KIND",
    "HAS PART",
    "HAS INSTANCE",
    "HAS MEMBER",
    "SUBSTANCE",
    "CAUSES",
    "DIFFERENT FROM",
    "VERB GROUP",
    "USAGE",
    "REGION",
]

# The cells of a candidate synset on the sheets of Dataset B, and the
# heading each one gets when they are joined into the full model input.
SYNSET_CELLS = {
    "BabelNet情報": None,
    "関係_日本語": "関係(日本語)",
    "IS A 2step_日本語": "IS A 2step(日本語)",
    "関係_英語": "関係(英語)",
    "IS A 2step_英語": "IS A 2step(英語)",
    "中国語BabelNet情報": None,
}


def record_text(record: pd.Series) -> str:
    """The WLSP cell: headword, co-classified words and classification.

    At most three co-classified words are shown. A missing division is
    printed as "nan", as it was on the sheets.
    """
    return "\n".join(
        [
            f"見出し: {record['headword']}",
            f"近義語: {'・'.join(record['co_classified'][:3])}",
            f"部門: {record['division']}",
            f"中項目: {record['section']}",
            f"分類項目: {record['category']}など",
            f"分類語彙表番号: {record['category_no']}",
        ]
    )


def headword_text(record: pd.Series) -> str:
    """The shortest WLSP input: the headword and its reading."""
    reading = record["reading"]
    text = f"見出し: {record['headword']}"
    return text + f"\n読み: {reading}" if isinstance(reading, str) and reading else text


def unique(values: Iterable[str]) -> list[str]:
    """Strip the values and drop empty ones and repeats, keeping the order."""
    seen: dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if text:
            seen.setdefault(text)
    return list(seen)


def _join(values: Iterable[str], n: int) -> str:
    return "・".join(unique(values)[:n])


def _get(synsets: Synsets, sid: str, field: str, lang: str) -> list[str]:
    return list(synsets.get(sid, {}).get(field, {}).get(lang, []))


def _lemmas_preferring(synsets: Synsets, sid: str, lang: str) -> list[str]:
    """The synset's first three lemmas in `lang`, or in the other language."""
    other = "EN" if lang == "JA" else "JA"
    lemmas = unique(_get(synsets, sid, "lemmas", lang))
    return (lemmas or unique(_get(synsets, sid, "lemmas", other)))[:3]


def _targets_text(synsets: Synsets, sid: str, label: str, lang: str) -> str:
    """The first three lemmas over all targets of one relation of `sid`."""
    targets = synsets.get(sid, {}).get("relations", {}).get(label, [])
    return _join(
        (lemma for t in targets for lemma in _lemmas_preferring(synsets, t, lang)), 3
    )


def _info_cell(synsets: Synsets, sid: str) -> str:
    lines = [
        f"日見出し語: {_join(_get(synsets, sid, 'lemmas', 'JA'), 3)}",
        f"英見出し語: {_join(_get(synsets, sid, 'lemmas', 'EN'), 3)}",
        f"日カテゴリー: {_join(_get(synsets, sid, 'categories', 'JA'), 3)}",
        "日説明文:",
        *unique(_get(synsets, sid, "glosses", "JA"))[:2],
        "英説明文:",
        *unique(_get(synsets, sid, "glosses", "EN"))[:2],
    ]
    return "\n".join(lines).strip() if sid in synsets else ""


def _relations_cell(synsets: Synsets, sid: str, lang: str) -> str:
    lines = []
    for label in RELATIONS:
        text = _targets_text(synsets, sid, label, lang)
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _is_a_2step_cell(synsets: Synsets, sid: str, lang: str) -> str:
    """The first three IS A targets, each followed by its own IS A targets."""
    lines = []
    for first in unique(synsets.get(sid, {}).get("relations", {}).get("IS A", []))[:3]:
        name = _join(_lemmas_preferring(synsets, first, lang), 1) or first
        second = _targets_text(synsets, first, "IS A", lang)
        lines.append(f"{name}: {second}" if second else f"{name}:")
    return "\n".join(lines)


def _chinese_cell(synsets: Synsets, sid: str) -> str:
    if sid not in synsets:
        return ""
    main = synsets[sid].get("main_gloss", {}).get("ZH")
    glosses = ([main] if main else []) + _get(synsets, sid, "glosses", "ZH")
    lines = [
        f"中国語見出し語: {_join(_get(synsets, sid, 'lemmas', 'ZH'), 3)}",
        f"中国語カテゴリー: {_join(_get(synsets, sid, 'categories', 'ZH'), 3)}",
        "中国語説明文:",
        *unique(glosses)[:2],
    ]
    return "\n".join(lines).strip()


def synset_cells(synsets: Synsets, sid: str) -> dict[str, str]:
    """The six cells that describe a candidate synset on the sheets of Dataset B."""
    return {
        "BabelNet情報": _info_cell(synsets, sid),
        "関係_日本語": _relations_cell(synsets, sid, "JA"),
        "IS A 2step_日本語": _is_a_2step_cell(synsets, sid, "JA"),
        "関係_英語": _relations_cell(synsets, sid, "EN"),
        "IS A 2step_英語": _is_a_2step_cell(synsets, sid, "EN"),
        "中国語BabelNet情報": _chinese_cell(synsets, sid),
    }


def joined_synset_cells(cells: Mapping[str, str]) -> str:
    """The six cells joined into one text, each non-empty one under its heading."""
    parts = []
    for name, heading in SYNSET_CELLS.items():
        text = cells[name].strip()
        if text:
            parts.append(f"{heading}:\n{text}" if heading else text)
    return "\n\n".join(parts)


def cell_field(text: str, label: str) -> str:
    """The value of the line of `text` that starts with `label` and a colon."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(label + ":"):
            return line[len(label) + 1 :].strip()
    return ""


def _listed(text: str, label: str) -> set[str]:
    return {t.strip() for t in cell_field(text, label).split("・") if t.strip()}


def surface_match(wlsp_full: str, bn_1step: str) -> bool:
    """Whether the record's headword or a co-classified word is a Japanese lemma.

    Only the words the texts show count: three co-classified words and
    three lemmas at most.
    """
    lemmas = _listed(bn_1step, "日見出し語")
    return cell_field(wlsp_full, "見出し") in lemmas or bool(
        _listed(wlsp_full, "近義語") & lemmas
    )


def synset_text_a(synsets: Synsets, sid: str) -> str:
    """The cell that described a candidate synset on the sheet of Dataset A.

    Lemmas, categories and hypernym lemmas are taken in Japanese, or in
    English when there are none in Japanese, and are not deduplicated; the
    cell is empty for a synset without lemmas.
    """

    def ja_or_en(s: str, field: str) -> list[str]:
        return (_get(synsets, s, field, "JA") or _get(synsets, s, field, "EN"))[:3]

    lemmas = "・".join(ja_or_en(sid, "lemmas"))
    if not lemmas:
        return ""
    hypernyms = synsets.get(sid, {}).get("hypernyms", [])
    lines = [
        f"単語: {lemmas}",
        f"\t\tカテゴリー: {'・'.join(ja_or_en(sid, 'categories'))}",
        f"上位概念: {'・'.join(x for h in hypernyms for x in ja_or_en(h, 'lemmas'))}",
        "説明文:",
        *_get(synsets, sid, "glosses", "JA")[:2],
        *_get(synsets, sid, "glosses", "EN")[:1],
    ]
    return "\n".join(lines)
