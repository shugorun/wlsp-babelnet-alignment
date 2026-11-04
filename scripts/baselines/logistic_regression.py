"""Logistic Regression: a logistic regression on ten scores, trained on the
development set.

The scores, in the order of FEATURES, are those of the other baselines and
variants of them: Headword Match and the Dice coefficient of the same
words; the Dice coefficients of the character n-grams of the joined
fields, of the classification and the categories, and of the joined WLSP
fields and the glosses; TF-IDF; the cosine similarity of the E5 vectors of
the joined fields and of the sentences of embeddings.py; the CSLS
similarity; and the rank in the record by the cosine of the sentences.

The n of each score with character n-grams is the one with which that score
alone reaches the highest F1 on the development set. The scores are
standardized with the means and standard deviations of the development
set, the model is trained there with C = 1, and the threshold on its
probability is the one with the highest F1 there (src/wlsp_bn/baselines.py).

    uv run --extra baselines python scripts/baselines/logistic_regression.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wlsp_bn import baselines as bl

# The scores with character n-grams, as functions of the pairs and n.
NGRAM_SCORES = {
    "char_dice": bl.char_dice,
    "category_dice": lambda p, n: bl.char_dice(p, n, "w_class", "b_category"),
    "gloss_dice": lambda p, n: bl.char_dice(p, n, "w_all", "b_gloss"),
    "tfidf": bl.tfidf,
}
FEATURES = [
    "headword_match",
    "lemma_dice",
    *NGRAM_SCORES,
    "e5_cosine_fields",
    "e5_cosine",
    "e5_csls",
    "e5_rank_in_record",
]


def features(pairs: pd.DataFrame, ngrams: dict[str, int]) -> np.ndarray:
    sentences, fields = bl.e5_vectors(pairs), bl.e5_vectors(pairs, sentences=False)
    columns = [bl.headword_match(pairs), bl.lemma_dice(pairs)]
    columns += [score(pairs, ngrams[name]) for name, score in NGRAM_SCORES.items()]
    columns += [
        bl.cosine(*fields),
        bl.cosine(*sentences),
        bl.csls(*sentences),
        bl.rank_in_record(pairs, bl.cosine(*sentences)),
    ]
    return np.stack(columns, axis=1)


def main() -> None:
    dev, test = bl.load_split("dev"), bl.load_split("test")
    ngrams = {
        name: bl.fit_ngram(dev["y"], lambda n, score=score: score(dev, n))[0]
        for name, score in NGRAM_SCORES.items()
    }
    x = features(dev, ngrams)
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit((x - mean) / std, dev["y"])
    threshold, dev_f1 = bl.best_threshold(
        dev["y"], model.predict_proba((x - mean) / std)[:, 1]
    )
    probability = model.predict_proba((features(test, ngrams) - mean) / std)[:, 1]

    settings = {
        "ngrams": ngrams,
        "threshold": threshold,
        "dev_f1": dev_f1,
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    bl.write_predictions("logistic-regression", test, probability > threshold, settings)


if __name__ == "__main__":
    main()
