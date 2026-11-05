# Baselines

These are the baselines of Table 2 (paper Section 5.3). Each is one script
in `scripts/baselines/`, which reads the model inputs of Dataset B
(`data/interim/pairs_b.parquet`) and its split (`data/splits/dataset_b.tsv`),
chooses its settings on the development set and writes its predictions for
the test set to `data/predictions/b-test/baselines/<method>/run1.tsv`, in the
form `scripts/evaluation/02_judge.py` writes a run; `run1.json` next to it
records the settings. The scores are in `src/wlsp_bn/baselines.py`.

| Row of Table 2 | Script | Predicted EQ |
|---|---|---|
| All-positive | `all_positive.py` | every pair |
| Headword Match | `headword_match.py` | the pairs whose WLSP words include a lemma of the synset |
| TF–IDF | `tfidf.py` | the pairs whose TF-IDF similarity is above a threshold |
| Multilingual Embeddings | `embeddings.py` | the pairs whose E5 CSLS similarity is above a threshold |
| Score Averaging | `score_averaging.py` | the pairs whose mean of three z-normalized scores is above a threshold |
| Logistic Regression | `logistic_regression.py` | the pairs whose probability under a logistic regression on ten scores is above a threshold |

The last three need the `baselines` extra, which installs PyTorch, and
download `intfloat/multilingual-e5-base` from Hugging Face on their first
run:

```
uv sync --locked --extra baselines
uv run python scripts/baselines/logistic_regression.py
```

## Input

The baselines read part of the text the models are given. On the WLSP side
they read the headword, the co-classified words (three at most) and the
classification: division, section and category. On the BabelNet side they
read the Japanese lemmas (three at most), the Japanese categories (three at
most) and the Japanese glosses (two at most). Where a score compares texts, the fields of each side
are joined into one text.

## Settings

A threshold is chosen among 200 quantiles of the scores of the development
set, as the one with the highest F1, the lowest on a tie. A score with
character n-grams takes the n from 1 to 3 with which it reaches the highest
F1 there. The test set is then scored with these settings unchanged.

The IDF of TF-IDF, the neighbours of CSLS, the rank of a candidate within
its record and the z-scores of Score Averaging are computed over the pairs
of the split being scored, without their labels.

## Methods

**Headword Match** predicts EQ when the WLSP headword or one of the
co-classified words is among the Japanese lemmas of the synset.

**TF–IDF** is the cosine similarity of the character n-gram TF-IDF vectors
of the two texts. An n-gram counts once in a text, and its IDF is
log((N + 1) / (df + 1)) + 1 over the N synset texts of the split. n = 1 was
chosen.

**Multilingual Embeddings** writes each side as Japanese sentences built
from its fields:

```
「<word>」は「<division> <section> <category>」のカテゴリに属する語である。同じカテゴリに属する語として「<other words>」が挙げられる。
「<lemma>」の定義文は「<glosses>」である。同義語として「<other lemmas>」が挙げられる。カテゴリは「<categories>」である。
```

The first word or lemma in code point order comes first. A sentence about
other words, lemmas or categories is left out when there are none, and a
synset without Japanese glosses starts with 「<lemma>」という語がある。 instead.
multilingual-e5-base embeds the WLSP sentence as a query and the BabelNet
sentence as a passage. The score is their CSLS similarity: the cosine
similarity, less half the mean similarity of the WLSP vector to its 10
nearest synset vectors and half that of the synset vector to its 10 nearest
WLSP vectors, among the distinct vectors of the split.

**Score Averaging** is the mean of three scores, each z-normalized: the
TF-IDF similarity, the CSLS similarity, and the rank of the candidate within
its record by the cosine similarity of the E5 vectors.

**Logistic Regression** is trained on the development set, with C = 1, on
ten scores standardized with the means and standard deviations of that
set:

- Headword Match, and the Dice coefficient of the WLSP words and the lemmas;
- the Dice coefficients of the character n-grams of the two texts, of the
  classification and the categories, and of the WLSP text and the glosses;
- the TF-IDF similarity;
- the cosine similarity of the E5 vectors of the two texts, and of the two
  sentences;
- the CSLS similarity, and the rank within the record by the cosine
  similarity of the sentences.

The threshold is on the probability the model gives.

## Comparison with the LLMs

`scripts/evaluation/03_score.py` scores the predictions and compares them
with those of the LLMs, counting each baseline as a condition with one run
([docs/evaluation.md](evaluation.md)). The difference in F1 of Section 6.1
between Qwen with the initial prompt and Logistic Regression, with its 95%
confidence interval:

```
uv run python scripts/evaluation/03_score.py data/predictions/b-test/qwen3.5-27b/v0 \
    data/predictions/b-test/baselines/logistic-regression --compare
```
