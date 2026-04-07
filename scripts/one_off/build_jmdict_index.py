# build_jmdict_index.py
# Build a simple Japanese-term -> English gloss list index from JMdict.

import xml.etree.ElementTree as ET
import re
import pickle
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SEP = re.compile(r"\s*(?:,|;|/|\bor\b)\s*")
MAX_WORDS = 3
BADLIST = {"asshole"}


def strip_parens(txt: str) -> str:
    """Remove parenthesized text and normalize spaces."""
    t = txt
    while True:
        new = re.sub(r"\([^()]*\)", " ", t)
        if new == t:
            break
        t = new
    return re.sub(r"\s+", " ", t).strip()


def extract_entry(el):
    """Extract Japanese keys and short English glosses from one JMdict entry."""
    keys = [k.text for k in el.findall("k_ele/keb") if k.text] + \
           [r.text for r in el.findall("r_ele/reb") if r.text]

    glosses = set()
    for s in el.findall("sense"):
        for g in s.findall("gloss"):
            if not g.text:
                continue

            t = strip_parens(g.text.strip())

            # Keep only short, simple glosses for indexing.
            if not t or len(t.split()) > MAX_WORDS or t.lower() in BADLIST:
                continue

            glosses.add(t)

            # Also split multi-part glosses such as "A, B / C".
            for p in SEP.split(t):
                if p.strip() and p.lower() not in BADLIST:
                    glosses.add(p.strip())

    return keys, sorted(glosses)


def build_index(jmdict_path: Path, out_path: Path) -> None:
    """Build and save a key -> gloss list dictionary."""
    mapping = {}

    # Stream the XML file entry by entry to avoid loading everything into memory.
    for _, el in ET.iterparse(jmdict_path, events=("end",)):
        if el.tag != "entry":
            continue

        keys, glosses = extract_entry(el)

        for k in keys:
            if glosses:
                mapping.setdefault(k, []).extend(glosses)

        el.clear()

    with open(out_path, "wb") as f:
        pickle.dump(mapping, f)


if __name__ == "__main__":
    jmdict_path = RAW_DIR / "JMdict_e.xml"
    out_path = PROCESSED_DIR / "JMdict.pkl"
    build_index(jmdict_path, out_path)

# Number of indexed keys: 402,803