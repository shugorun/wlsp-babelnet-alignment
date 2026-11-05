"""TF-IDF: the cosine similarity of the character n-gram TF-IDF vectors of
the WLSP and the BabelNet side, each its fields joined into one text.

n (1 to 3) and the threshold are those with the highest F1 on the
development set (src/wlsp_bn/baselines.py).

    uv run python scripts/baselines/tfidf.py
"""

from wlsp_bn import baselines as bl


def main() -> None:
    dev, test = bl.load_split("dev"), bl.load_split("test")
    n, threshold, dev_f1 = bl.fit_ngram(dev["y"], lambda n: bl.tfidf(dev, n))
    settings = {"ngram": n, "threshold": threshold, "dev_f1": dev_f1}
    bl.write_predictions("tfidf", test, bl.tfidf(test, n) > threshold, settings)


if __name__ == "__main__":
    main()
