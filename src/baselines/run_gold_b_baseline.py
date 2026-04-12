#!/usr/bin/env python
# Usage: python src/baselines/run_gold_b_baseline.py --records data/gold/gold_B_records.pkl --gold data/gold/gold_B.pkl --babelnet data/processed/babelnet_.pkl --output-dir outputs/baselines/gold_B
"""Build a simple gold_B baseline from sids_JA and sids_JMdict only.

This script reads only pickle files, builds pairwise features for
record_id x synset_id, runs grouped cross-validation on gold_B, and saves
the feature table, out-of-fold predictions, summary metrics, and a final
trained logistic-regression model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.experiment_log import append_run_log

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RECORDS_PATH = ROOT / "data" / "gold" / "gold_B_records.pkl"
DEFAULT_GOLD_PATH = ROOT / "data" / "gold" / "gold_B.pkl"
DEFAULT_BABELNET_PATH = ROOT / "data" / "processed" / "babelnet_.pkl"

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B"

RANDOM_STATE = 0
N_SPLITS = 5
POSITIVE_LABEL = "EQUAL"

FEATURE_COLUMNS = [
    "from_sids_JA",
    "from_sids_JMdict",
    "source_count",
    "has_bilingual_support",
    "lemma_exact_in_lemmas_JA",
    "lemma_substring_in_lemmas_JA",
    "max_ja_char_jaccard",
    "jmdict_exact_in_lemmas_EN",
    "jmdict_substring_in_lemmas_EN",
    "max_en_token_jaccard",
    "ja_category_overlap_count",
    "main_gloss_ja_char_jaccard",
    "main_gloss_en_token_jaccard",
    "num_lemmas_JA",
    "num_lemmas_EN",
    "num_categories_JA",
]

# Hand-tuned score used as the non-learned reference baseline.
WEIGHTED_SCORE_WEIGHTS = {
    "from_sids_JA": 2.0,
    "from_sids_JMdict": 2.0,
    "source_count": 0.8,
    "has_bilingual_support": 1.2,
    "lemma_exact_in_lemmas_JA": 3.0,
    "lemma_substring_in_lemmas_JA": 0.7,
    "max_ja_char_jaccard": 1.5,
    "jmdict_exact_in_lemmas_EN": 3.0,
    "jmdict_substring_in_lemmas_EN": 0.7,
    "max_en_token_jaccard": 1.5,
    "ja_category_overlap_count": 0.4,
    "main_gloss_ja_char_jaccard": 0.8,
    "main_gloss_en_token_jaccard": 0.8,
    "num_lemmas_JA": -0.05,
    "num_lemmas_EN": -0.03,
    "num_categories_JA": -0.05,
}


def path_for_log(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_ja_text(text: object) -> str:
    """Normalize one Japanese-like text cell."""
    if text is None or pd.isna(text):
        return ""
    out = str(text).strip().lower()
    out = re.sub(r"\s+", "", out)
    return out


def normalize_en_text(text: object) -> str:
    """Normalize one English-like text cell."""
    if text is None or pd.isna(text):
        return ""
    out = str(text).strip().lower()
    out = out.replace("_", " ")
    out = re.sub(r"[^a-z0-9 ]+", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def normalize_list(value: object, normalizer) -> list[str]:
    """Convert one cell to a clean list of normalized strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif not isinstance(value, (list, tuple, set, pd.Series, pd.Index)):
        value = [value]

    out = []
    for item in value:
        if item is None or pd.isna(item):
            continue
        text = normalizer(item)
        if text:
            out.append(text)
    return out


def char_bigrams(text: str) -> set[str]:
    """Return character bigrams for a normalized string."""
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[i : i + 2] for i in range(len(text) - 1)}


def token_set(text: str) -> set[str]:
    """Return a token set for a normalized English string."""
    if not text:
        return set()
    return {token for token in text.split(" ") if token}


def jaccard_score(left: set[str], right: set[str]) -> float:
    """Return Jaccard similarity for two sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def max_char_jaccard(query: str, candidates: list[str]) -> float:
    """Return the max character-bigram Jaccard score."""
    query_bigrams = char_bigrams(query)
    return max((jaccard_score(query_bigrams, char_bigrams(item)) for item in candidates), default=0.0)


def max_token_jaccard(queries: list[str], candidates: list[str]) -> float:
    """Return the max token-level Jaccard score."""
    scores = []
    for query in queries:
        query_tokens = token_set(query)
        for candidate in candidates:
            scores.append(jaccard_score(query_tokens, token_set(candidate)))
    return max(scores, default=0.0)


def build_record_context(record: pd.Series) -> tuple[str, list[str], set[str]]:
    """Build normalized record-level text helpers."""
    ja_parts = [
        record.get("division", ""),
        record.get("category", ""),
        record.get("subcategory", ""),
        record.get("class", ""),
        record.get("lemma", ""),
        record.get("kana", ""),
    ]
    en_terms = normalize_list(record.get("EN_JMdict", []), normalize_en_text)
    ja_text = normalize_ja_text("".join(str(part) for part in ja_parts))
    ja_categories = {
        normalize_ja_text(record.get("division", "")),
        normalize_ja_text(record.get("category", "")),
        normalize_ja_text(record.get("subcategory", "")),
        normalize_ja_text(record.get("class", "")),
    }
    ja_categories.discard("")
    return ja_text, en_terms, ja_categories


def build_feature_rows(
    records: pd.DataFrame,
    gold: pd.DataFrame,
    babelnet: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feature row per gold_B pair."""
    rows = []

    # gold_B already defines the candidate pairs to score, so this step only
    # materializes features for each record_id x synset_id pair.
    for pair in gold.itertuples(index=False):
        record_id = int(pair.record_id)
        synset_id = str(pair.synset_id)
        record = records.loc[record_id]
        synset = babelnet.loc[synset_id]

        sids_ja = {str(sid) for sid in normalize_list(record.get("sids_JA", []), str)}
        sids_jmdict = {str(sid) for sid in normalize_list(record.get("sids_JMdict", []), str)}

        lemma = normalize_ja_text(record.get("lemma", ""))
        en_terms = normalize_list(record.get("EN_JMdict", []), normalize_en_text)
        record_text_ja, record_terms_en, record_categories_ja = build_record_context(record)

        lemmas_ja = normalize_list(synset.get("lemmas_JA", []), normalize_ja_text)
        lemmas_en = normalize_list(synset.get("lemmas_EN", []), normalize_en_text)
        categories_ja = normalize_list(synset.get("categories_JA", []), normalize_ja_text)
        main_gloss_ja = normalize_ja_text(synset.get("main_gloss_JA", ""))
        main_gloss_en = normalize_en_text(synset.get("main_gloss_EN", ""))

        row = {
            "record_id": record_id,
            "synset_id": synset_id,
            "label": 1 if pair.label == POSITIVE_LABEL else 0,
            "raw_label": str(pair.label),
            "from_sids_JA": int(synset_id in sids_ja),
            "from_sids_JMdict": int(synset_id in sids_jmdict),
            "lemma_exact_in_lemmas_JA": int(lemma in set(lemmas_ja)),
            "lemma_substring_in_lemmas_JA": int(any(lemma and lemma in item for item in lemmas_ja)),
            "max_ja_char_jaccard": max_char_jaccard(lemma, lemmas_ja),
            "jmdict_exact_in_lemmas_EN": int(any(term in set(lemmas_en) for term in en_terms)),
            "jmdict_substring_in_lemmas_EN": int(
                any(term and term in item for term in en_terms for item in lemmas_en)
            ),
            "max_en_token_jaccard": max_token_jaccard(record_terms_en, lemmas_en),
            "ja_category_overlap_count": int(len(record_categories_ja & set(categories_ja))),
            "main_gloss_ja_char_jaccard": max_char_jaccard(record_text_ja, [main_gloss_ja]),
            "main_gloss_en_token_jaccard": max_token_jaccard(record_terms_en, [main_gloss_en]),
            "num_lemmas_JA": len(lemmas_ja),
            "num_lemmas_EN": len(lemmas_en),
            "num_categories_JA": len(categories_ja),
        }
        row["source_count"] = row["from_sids_JA"] + row["from_sids_JMdict"]
        row["has_bilingual_support"] = int(row["source_count"] == 2)
        rows.append(row)

    return pd.DataFrame(rows)


def weighted_score(df: pd.DataFrame) -> np.ndarray:
    """Return the weighted baseline score."""
    score = np.zeros(len(df), dtype=float)
    for col, weight in WEIGHTED_SCORE_WEIGHTS.items():
        score += df[col].to_numpy(dtype=float) * weight
    return score


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select the threshold that maximizes F1 on one score vector."""
    candidates = np.unique(scores)
    if len(candidates) == 0:
        return 0.0

    best_t = float(candidates[0])
    best_f1 = -1.0

    for threshold in candidates:
        y_pred = scores >= threshold
        metrics = binary_metrics(y_true, y_pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(threshold)

    return best_t


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return binary classification metrics."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def record_level_metrics(
    df: pd.DataFrame,
    pred_col: str,
) -> dict[str, float]:
    """Return record-level set metrics."""
    precisions = []
    recalls = []
    f1s = []
    exact_matches = []

    for _, group in df.groupby("record_id", sort=False):
        gold_set = set(group.loc[group["label"] == 1, "synset_id"])
        pred_set = set(group.loc[group[pred_col] == 1, "synset_id"])

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        exact_matches.append(float(gold_set == pred_set))

    return {
        "record_macro_precision": float(np.mean(precisions)),
        "record_macro_recall": float(np.mean(recalls)),
        "record_macro_f1": float(np.mean(f1s)),
        "record_exact_match_rate": float(np.mean(exact_matches)),
    }


def run_grouped_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped cross-validation for weighted score and logistic regression."""
    groups = df["record_id"].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    gkf = GroupKFold(n_splits=N_SPLITS)
    out = df[["record_id", "synset_id", "label", "raw_label"]].copy()
    out["fold"] = -1
    out["weighted_score"] = np.nan
    out["weighted_pred"] = 0
    out["weighted_threshold"] = np.nan
    out["logreg_score"] = np.nan
    out["logreg_pred"] = 0
    out["logreg_threshold"] = np.nan

    fold_rows = []

    # GroupKFold keeps pairs from the same record in the same split.
    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_weighted_score = weighted_score(train_df)
        test_weighted_score = weighted_score(test_df)
        weighted_t = best_threshold(y_train, train_weighted_score)
        weighted_pred = (test_weighted_score >= weighted_t).astype(int)

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        train_logreg_score = model.predict_proba(X_train)[:, 1]
        test_logreg_score = model.predict_proba(X_test)[:, 1]
        logreg_t = best_threshold(y_train, train_logreg_score)
        logreg_pred = (test_logreg_score >= logreg_t).astype(int)

        out.loc[out.index[test_idx], "fold"] = fold_id
        out.loc[out.index[test_idx], "weighted_score"] = test_weighted_score
        out.loc[out.index[test_idx], "weighted_pred"] = weighted_pred
        out.loc[out.index[test_idx], "weighted_threshold"] = weighted_t
        out.loc[out.index[test_idx], "logreg_score"] = test_logreg_score
        out.loc[out.index[test_idx], "logreg_pred"] = logreg_pred
        out.loc[out.index[test_idx], "logreg_threshold"] = logreg_t

        weighted_metrics = binary_metrics(y[test_idx], weighted_pred)
        logreg_metrics = binary_metrics(y[test_idx], logreg_pred)

        fold_rows.append(
            {
                "fold": fold_id,
                "n_train_pairs": len(train_idx),
                "n_test_pairs": len(test_idx),
                "weighted_threshold": weighted_t,
                "weighted_precision": weighted_metrics["precision"],
                "weighted_recall": weighted_metrics["recall"],
                "weighted_f1": weighted_metrics["f1"],
                "logreg_threshold": logreg_t,
                "logreg_precision": logreg_metrics["precision"],
                "logreg_recall": logreg_metrics["recall"],
                "logreg_f1": logreg_metrics["f1"],
            }
        )

    return out, pd.DataFrame(fold_rows)


def summarize_results(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact metrics table."""
    rows = []

    for method in ["weighted", "logreg"]:
        pair_metrics = binary_metrics(
            oof["label"].to_numpy(dtype=int),
            oof[f"{method}_pred"].to_numpy(dtype=int),
        )
        record_metrics = record_level_metrics(oof, f"{method}_pred")

        row = {
            "method": method,
            "pair_precision": pair_metrics["precision"],
            "pair_recall": pair_metrics["recall"],
            "pair_f1": pair_metrics["f1"],
            "pair_accuracy": pair_metrics["accuracy"],
            "record_macro_precision": record_metrics["record_macro_precision"],
            "record_macro_recall": record_metrics["record_macro_recall"],
            "record_macro_f1": record_metrics["record_macro_f1"],
            "record_exact_match_rate": record_metrics["record_exact_match_rate"],
            "mean_fold_f1": float(fold_metrics[f"{method}_f1"].mean()),
            "std_fold_f1": float(fold_metrics[f"{method}_f1"].std(ddof=0)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def train_final_model(df: pd.DataFrame) -> dict[str, object]:
    """Train the final logistic-regression model on all gold_B pairs."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)

    full_score = model.predict_proba(X)[:, 1]
    threshold = best_threshold(y, full_score)

    return {
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "model": model,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    ap.add_argument("--babelnet", type=Path, default=DEFAULT_BABELNET_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    features_path = output_dir / "pair_features.pkl"
    oof_path = output_dir / "oof_predictions.pkl"
    metrics_path = output_dir / "metrics.pkl"
    model_path = output_dir / "logreg_model.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the three core tables: WLSP records, gold labels, and BabelNet rows.
    records = pd.read_pickle(args.records)
    gold = pd.read_pickle(args.gold)
    babelnet = pd.read_pickle(args.babelnet)
    babelnet.index = babelnet.index.map(str)

    # Export both intermediate features and evaluation outputs so later runs
    # can reuse them as the lexical baseline reference point.
    feature_df = build_feature_rows(records, gold, babelnet)
    feature_df.to_pickle(features_path)
    print(f"Saved features: {features_path}")

    oof_df, fold_metrics = run_grouped_cv(feature_df)
    oof_df.to_pickle(oof_path)
    print(f"Saved OOF predictions: {oof_path}")

    metrics_df = summarize_results(oof_df, fold_metrics)
    metrics_obj = {
        "summary": metrics_df,
        "folds": fold_metrics,
    }
    with metrics_path.open("wb") as f:
        pickle.dump(metrics_obj, f)
    print(f"Saved metrics: {metrics_path}")

    final_model = train_final_model(feature_df)
    with model_path.open("wb") as f:
        pickle.dump(final_model, f)
    print(f"Saved final model: {model_path}")

    print(metrics_df.to_string(index=False))

    summary_row = metrics_df.iloc[0].to_dict()
    append_run_log(
        run_name="gold_B_pairwise_baseline",
        rationale="Rebuild the lexical / pairwise baseline on gold_B as the reference point for later ranking and hybrid runs.",
        script_path="src/baselines/run_gold_b_baseline.py",
        params={
            "model_weighted": "rule_weighted_scoring",
            "model_logreg": "LogisticRegression",
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
        },
        metrics={
            "weighted_pair_f1": summary_row["pair_f1"] if summary_row["method"] == "weighted" else metrics_df.loc[metrics_df["method"] == "weighted", "pair_f1"].iloc[0],
            "logreg_pair_f1": metrics_df.loc[metrics_df["method"] == "logreg", "pair_f1"].iloc[0],
            "logreg_record_exact_match_rate": metrics_df.loc[metrics_df["method"] == "logreg", "record_exact_match_rate"].iloc[0],
        },
        outputs=[
            path_for_log(features_path),
            path_for_log(oof_path),
            path_for_log(metrics_path),
            path_for_log(model_path),
        ],
    )


if __name__ == "__main__":
    main()
