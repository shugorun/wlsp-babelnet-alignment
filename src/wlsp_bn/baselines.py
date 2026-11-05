"""The scores of the baselines of Table 2 and the choice of their settings.

The baselines read the same text as the models (data/interim/pairs_b.parquet),
but only some of its fields: on the WLSP side the headword, the co-classified
words and the classification (division, section and category), and on the
BabelNet side the Japanese lemmas, categories and glosses. Each baseline
gives every pair of a split a score, and the pairs whose score is above a
threshold are predicted EQ. The threshold, and the n of the character
n-grams for the scores that use them, are those with the highest F1 on the
development set, and the test set is scored with them unchanged.

Some scores also depend on the other pairs of the split being scored,
though not on their labels: the IDF of TF-IDF, the neighbours of CSLS, the
rank of a candidate within its record and the z-scores of Score Averaging.

The E5 scores need the `baselines` extra (`uv sync --extra baselines`).
"""

import json
from collections import Counter
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np
import pandas as pd

from wlsp_bn import paths
from wlsp_bn.cells import surface_match
from wlsp_bn.metrics import precision_recall_f1

NGRAMS = (1, 2, 3)
N_THRESHOLDS = 200  # the thresholds tried are this many quantiles of the scores
E5_MODEL = "intfloat/multilingual-e5-base"
CSLS_NEIGHBOURS = 10

# The labels that start the fields of the texts. A field runs to the next
# line that starts with one of them, since the glosses are on the lines
# after their label.
LABELS = {
    "見出し",
    "近義語",
    "部門",
    "中項目",
    "分類項目",
    "分類語彙表番号",
    "読み",
    "日見出し語",
    "英見出し語",
    "日カテゴリー",
    "日説明文",
    "英説明文",
}
WLSP_CLASS = ["部門", "中項目", "分類項目"]
WLSP_ALL = ["見出し", "近義語", *WLSP_CLASS]
BN_ALL = ["日見出し語", "日カテゴリー", "日説明文"]


def field(text: str, label: str) -> str:
    """The value of the field `label` of `text`, its lines joined by spaces."""
    out, inside = [], False
    for line in str(text).split("\n"):
        line = line.strip()
        name, colon, value = line.partition(":")
        if colon and name.strip() in LABELS:
            inside = name.strip() == label
            if inside and value.strip():
                out.append(value.strip())
        elif inside and line:
            out.append(line)
    return " ".join(out)


def _words(text: str) -> set[str]:
    return {t.strip() for t in text.split("・") if t.strip()}


def load_split(split: str) -> pd.DataFrame:
    """The pairs of one split of Dataset B, with the fields the baselines read.

    Adds y (whether the pair is EQ), the WLSP fields w_words (the headword
    and co-classified words), w_class and w_all, and the BabelNet fields
    b_lemmas, b_category, b_gloss and b_all; *_all joins the fields of its
    side.
    """
    pairs = pd.read_parquet(paths.INTERIM / "pairs_b.parquet")
    splits = pd.read_csv(paths.SPLITS / "dataset_b.tsv", sep="\t")
    chosen = splits.loc[splits["split"] == split, "record_id"]
    pairs = pairs[pairs["record_id"].isin(chosen)].reset_index(drop=True)
    w, b = pairs["wlsp_full"], pairs["bn_1step"]
    pairs["w_words"] = [
        ({h} if h else set()) | _words(field(t, "近義語"))
        for h, t in zip(w.map(lambda t: field(t, "見出し")), w, strict=True)
    ]
    pairs["w_class"] = w.map(lambda t: " ".join(field(t, k) for k in WLSP_CLASS))
    pairs["w_all"] = w.map(lambda t: " ".join(field(t, k) for k in WLSP_ALL))
    pairs["b_lemmas"] = b.map(lambda t: _words(field(t, "日見出し語")))
    pairs["b_category"] = b.map(lambda t: field(t, "日カテゴリー"))
    pairs["b_gloss"] = b.map(lambda t: field(t, "日説明文"))
    pairs["b_all"] = b.map(lambda t: " ".join(field(t, k) for k in BN_ALL))
    pairs["y"] = pairs["label"] == "EQ"
    return pairs


# ---------------------------------------------------------------- surface


def _dice(a: set, b: set) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 0.0


def _ngrams(text: str, n: int) -> set[str]:
    t = text.replace(" ", "")
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def headword_match(pairs: pd.DataFrame) -> np.ndarray:
    """1 if the pair has a surface match (src/wlsp_bn/cells.py), else 0."""
    sides = zip(pairs["wlsp_full"], pairs["bn_1step"], strict=True)
    return np.array([1.0 if surface_match(w, b) else 0.0 for w, b in sides])


def lemma_dice(pairs: pd.DataFrame) -> np.ndarray:
    """The Dice coefficient of the WLSP words and the lemmas of the synset."""
    sides = zip(pairs["w_words"], pairs["b_lemmas"], strict=True)
    return np.array([_dice(w, b) for w, b in sides])


def char_dice(
    pairs: pd.DataFrame, n: int, wlsp: str = "w_all", bn: str = "b_all"
) -> np.ndarray:
    """The Dice coefficient of the character n-grams of two fields."""
    sides = zip(pairs[wlsp], pairs[bn], strict=True)
    return np.array([_dice(_ngrams(w, n), _ngrams(b, n)) for w, b in sides])


def tfidf(pairs: pd.DataFrame, n: int) -> np.ndarray:
    """The cosine similarity of the character n-gram TF-IDF vectors of the sides.

    An n-gram counts once in a text, and its IDF, log((N + 1) / (df + 1)) + 1,
    is estimated from the N synset texts of the split.
    """
    docs_w = [_ngrams(t, n) for t in pairs["w_all"]]
    docs_b = [_ngrams(t, n) for t in pairs["b_all"]]
    df = Counter(g for d in docs_b for g in d)
    idf = {g: np.log((len(docs_b) + 1) / (c + 1)) + 1.0 for g, c in df.items()}
    out = []
    for a, b in zip(docs_w, docs_b, strict=True):
        common = a & b
        if not common:
            out.append(0.0)
            continue
        num = sum(idf.get(g, 0.0) ** 2 for g in common)
        na = np.sqrt(sum(idf.get(g, 0.0) ** 2 for g in a))
        nb = np.sqrt(sum(idf.get(g, 0.0) ** 2 for g in b))
        out.append(num / (na * nb) if na and nb else 0.0)
    return np.array(out)


# ---------------------------------------------------------------- E5


def wlsp_sentence(words: set[str], w_class: str) -> str:
    """The WLSP side as a sentence, naming first the first word in code point order."""
    head = sorted(words)[0] if words else ""
    others = "・".join(sorted(words - {head})) if len(words) > 1 else ""
    text = f"「{head}」は「{w_class.strip()}」のカテゴリに属する語である。"
    if others:
        text += f"同じカテゴリに属する語として「{others}」が挙げられる。"
    return text


def bn_sentence(lemmas: set[str], category: str, gloss: str) -> str:
    """The BabelNet side as a sentence, in the same way as wlsp_sentence."""
    head = sorted(lemmas)[0] if lemmas else ""
    gloss = gloss.strip()
    if gloss:
        text = f"「{head}」の定義文は「{gloss}」である。"
    else:
        text = f"「{head}」という語がある。"
    others = sorted(lemmas - {head})
    if others:
        text += f"同義語として「{'・'.join(others)}」が挙げられる。"
    if category.strip():
        text += f"カテゴリは「{category.strip()}」である。"
    return text


@cache
def _e5():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(E5_MODEL)


def _encode(texts: Sequence[str]) -> np.ndarray:
    unique = list(dict.fromkeys(texts))
    vectors = _e5().encode(
        unique, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    )
    index = {t: i for i, t in enumerate(unique)}
    return np.array([vectors[index[t]] for t in texts])


def e5_vectors(
    pairs: pd.DataFrame, sentences: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """The E5 vectors of the WLSP side, as queries, and of the BabelNet side.

    The sides are given as the sentences of wlsp_sentence and bn_sentence,
    or with `sentences` false as their joined fields.
    """
    if sentences:
        w_fields = zip(pairs["w_words"], pairs["w_class"], strict=True)
        b_fields = zip(
            pairs["b_lemmas"], pairs["b_category"], pairs["b_gloss"], strict=True
        )
        w = [wlsp_sentence(*f) for f in w_fields]
        b = [bn_sentence(*f) for f in b_fields]
    else:
        w, b = list(pairs["w_all"]), list(pairs["b_all"])
    return _encode(["query: " + t for t in w]), _encode(["passage: " + t for t in b])


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1)


def csls(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross-domain similarity local scaling of the pairs of rows of `a` and `b`.

    The cosine similarity, less half the mean similarity of the WLSP vector
    to its CSLS_NEIGHBOURS nearest synset vectors and half that of the synset
    vector to its CSLS_NEIGHBOURS nearest WLSP vectors, among the distinct
    vectors of the split. A vector that is close to everything is discounted.
    """
    ua, ia = np.unique(a, axis=0, return_inverse=True)
    ub, ib = np.unique(b, axis=0, return_inverse=True)
    k = min(CSLS_NEIGHBOURS, len(ub) - 1, len(ua) - 1)
    ra = np.sort(ua @ ub.T, axis=1)[:, -k:].mean(1)
    rb = np.sort(ub @ ua.T, axis=1)[:, -k:].mean(1)
    return cosine(a, b) - ra[ia] / 2 - rb[ib] / 2


# ---------------------------------------------------------------- combining


def rank_in_record(pairs: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Minus the rank of each pair by `scores` among the pairs of its record.

    Pairs with the same score are ranked in the order of the pairs.
    """
    out = np.zeros(len(pairs))
    records = pairs["record_id"].to_numpy()
    for r in np.unique(records):
        at = np.where(records == r)[0]
        ranks = np.empty(len(at), int)
        ranks[np.argsort(-scores[at], kind="stable")] = np.arange(len(at))
        out[at] = -ranks
    return out


def zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0.0


# ---------------------------------------------------------------- fitting


def best_threshold(y: Sequence[bool], scores: np.ndarray) -> tuple[float, float]:
    """The threshold with the highest F1, and that F1.

    The thresholds tried are N_THRESHOLDS quantiles of `scores`, and on a
    tie the lowest wins.
    """
    best_t, best_f1 = 0.0, -1.0
    for t in np.unique(np.quantile(scores, np.linspace(0, 1, N_THRESHOLDS))):
        f1 = precision_recall_f1(list(y), list(scores > t))[2]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def fit_ngram(
    y: Sequence[bool], scores: Callable[[int], np.ndarray]
) -> tuple[int, float, float]:
    """The n of NGRAMS whose `scores(n)` reach the highest F1, its threshold and F1.

    On a tie the smallest n wins.
    """
    fits = [(n, *best_threshold(y, scores(n))) for n in NGRAMS]
    return max(fits, key=lambda f: f[2])


def write_predictions(
    method: str, pairs: pd.DataFrame, predicted: np.ndarray, settings: dict
) -> None:
    """Write the predictions for the test set as 02_judge.py writes those of a run.

    They go to data/predictions/b-test/baselines/<method>/run1.tsv, and the
    settings chosen on the development set to run1.json next to it.
    """
    out_dir = paths.PREDICTIONS / "b-test" / "baselines" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "record_id": pairs["record_id"],
            "synset_id": pairs["synset_id"],
            "prediction": np.where(predicted, "EQUAL", "NONE"),
        }
    )
    out = out_dir / "run1.tsv"
    predictions.to_csv(out, sep="\t", index=False, lineterminator="\n")
    out.with_suffix(".json").write_text(
        json.dumps({"method": method, **settings}, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{out}: {len(predictions):,} pairs, {int(predicted.sum())} EQUAL")
