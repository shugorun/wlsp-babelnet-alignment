"""Score Averaging: the mean of three z-normalized scores.

The scores are the TF-IDF similarity of tfidf.py, with the n chosen there;
the CSLS similarity of embeddings.py; and the rank of the pair within its
record by the cosine similarity of the same E5 vectors. Each is
z-normalized over the pairs of the split. The threshold is the one with the
highest F1 on the development set (src/wlsp_bn/baselines.py).

    uv run --extra baselines python scripts/baselines/score_averaging.py
"""

import numpy as np
import pandas as pd

from wlsp_bn import baselines as bl


def mean_score(pairs: pd.DataFrame, n: int) -> np.ndarray:
    a, b = bl.e5_vectors(pairs)
    scores = [
        bl.tfidf(pairs, n),
        bl.csls(a, b),
        bl.rank_in_record(pairs, bl.cosine(a, b)),
    ]
    return np.mean([bl.zscore(s) for s in scores], axis=0)


def main() -> None:
    dev, test = bl.load_split("dev"), bl.load_split("test")
    n, _, _ = bl.fit_ngram(dev["y"], lambda n: bl.tfidf(dev, n))
    threshold, dev_f1 = bl.best_threshold(dev["y"], mean_score(dev, n))
    settings = {"ngram": n, "threshold": threshold, "dev_f1": dev_f1}
    predicted = mean_score(test, n) > threshold
    bl.write_predictions("score-averaging", test, predicted, settings)


if __name__ == "__main__":
    main()
