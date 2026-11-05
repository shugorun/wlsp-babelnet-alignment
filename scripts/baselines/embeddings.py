"""Multilingual Embeddings: the CSLS similarity of the multilingual-e5-base
vectors of the WLSP and the BabelNet side.

Each side is written as a few Japanese sentences built from its fields
and embedded as an E5 query (WLSP) or passage (BabelNet). CSLS discounts
the cosine similarity of vectors that are close to many others. The
threshold is the one with the highest F1 on the development set
(src/wlsp_bn/baselines.py).

    uv run --extra baselines python scripts/baselines/embeddings.py
"""

from wlsp_bn import baselines as bl


def main() -> None:
    dev, test = bl.load_split("dev"), bl.load_split("test")
    threshold, dev_f1 = bl.best_threshold(dev["y"], bl.csls(*bl.e5_vectors(dev)))
    settings = {"model": bl.E5_MODEL, "threshold": threshold, "dev_f1": dev_f1}
    predicted = bl.csls(*bl.e5_vectors(test)) > threshold
    bl.write_predictions("embeddings", test, predicted, settings)


if __name__ == "__main__":
    main()
