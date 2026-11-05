"""Reading the WLSP database and looking up its headwords in JMdict.

The columns are named after the levels of the WLSP classification as the
paper names them: class (類), division (部門), section (中項目) and category
(分類項目). Only records of the Nominal class (体の類) are used.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# Columns of bunruidb-fam.csv that are read, and their names here.
CSV_COLUMNS = {
    "レコードＩＤ番号": "record_id",
    "類": "class",
    "部門": "division",
    "中項目": "section",
    "分類項目": "category",
    "分類番号": "category_no",
    "段落番号": "paragraph_no",
    "小段落番号": "subparagraph_no",
    "語番号": "word_no",
    "見出し本体": "headword",
    "読み": "reading",
}
NOMINAL = "体"
# The database marks some rows with this headword; they are not entries.
NOT_AN_ENTRY = "＊"  # noqa: RUF001 (the full-width asterisk of the database)


def read_nominal_records(csv_path: Path) -> pd.DataFrame:
    """Return the Nominal records of the WLSP database, indexed by record_id.

    The classification number is read as a float, as it was when the
    datasets were built: its printed form and its number of decimal digits
    (used for the strata of Dataset A) are those of the float.
    """
    df = pd.read_csv(
        csv_path,
        usecols=list(CSV_COLUMNS),
        dtype=str,
        na_values=["", "NA", "N/A"],
        keep_default_na=True,
    ).rename(columns=CSV_COLUMNS)
    df["category_no"] = pd.to_numeric(df["category_no"], errors="coerce")
    for col in ["record_id", "paragraph_no", "subparagraph_no", "word_no"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    df = df[(df["class"] == NOMINAL) & (df["headword"] != NOT_AN_ENTRY)]
    return df.set_index("record_id")


def co_classified_words(records: pd.DataFrame) -> pd.Series:
    """Return, for each record, the other headwords of its subparagraph.

    These are the words WLSP classifies together with the record (the
    paper's co-classified words), in the order of the database, without
    the record's own headword.
    """
    key = ["category_no", "paragraph_no", "subparagraph_no"]
    groups = records.groupby(key)["headword"].apply(list).to_dict()
    return pd.Series(
        [
            [w for w in groups.get(tuple(row), []) if w != headword]
            for row, headword in zip(
                records[key].itertuples(index=False, name=None),
                records["headword"],
                strict=True,
            )
        ],
        index=records.index,
        name="co_classified",
    )


# Separators that split one gloss into several English terms.
_GLOSS_PARTS = re.compile(r"\s*(?:,|;|/|\bor\b)\s*")
_MAX_GLOSS_WORDS = 3
_EXCLUDED_GLOSSES = {"asshole"}


def _strip_parentheses(text: str) -> str:
    while True:
        stripped = re.sub(r"\([^()]*\)", " ", text)
        if stripped == text:
            return re.sub(r"\s+", " ", text).strip()
        text = stripped


def jmdict_index(xml_path: Path) -> dict[str, list[str]]:
    """Map each Japanese form in JMdict to the English terms of its entries.

    Every kanji and kana form of an entry maps to the entry's glosses of at
    most three words, with parenthesised text removed, and to the parts of
    those glosses split at commas, semicolons, slashes and "or". A form that
    belongs to several entries gets the terms of each entry in turn, so the
    list can hold a term more than once.
    """
    index: dict[str, list[str]] = {}
    for _, entry in ET.iterparse(xml_path, events=("end",)):
        if entry.tag != "entry":
            continue
        forms = [k.text for k in entry.findall("k_ele/keb") if k.text]
        forms += [r.text for r in entry.findall("r_ele/reb") if r.text]
        terms = set()
        for gloss in entry.findall("sense/gloss"):
            if not gloss.text:
                continue
            text = _strip_parentheses(gloss.text.strip())
            if (
                not text
                or len(text.split()) > _MAX_GLOSS_WORDS
                or text.lower() in _EXCLUDED_GLOSSES
            ):
                continue
            terms.add(text)
            for part in _GLOSS_PARTS.split(text):
                if part.strip() and part.lower() not in _EXCLUDED_GLOSSES:
                    terms.add(part.strip())
        if terms:
            for form in forms:
                index.setdefault(form, []).extend(sorted(terms))
        entry.clear()
    return index
