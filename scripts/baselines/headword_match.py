"""Headword Match: a pair is EQ if the WLSP headword or a co-classified word
is a Japanese lemma of the synset.

Only the words the texts show are compared: three co-classified words and
three lemmas at most (src/wlsp_bn/cells.py).

    uv run python scripts/baselines/headword_match.py
"""

from wlsp_bn import baselines as bl


def main() -> None:
    test = bl.load_split("test")
    bl.write_predictions("headword-match", test, bl.headword_match(test) > 0, {})


if __name__ == "__main__":
    main()
