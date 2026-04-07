"""Combine pairwise features with ranking features on gold_B.

This script loads the existing gold_B pairwise feature table and the
gold_B ranking outputs, merges them into one feature table, runs grouped
cross-validation, and saves the hybrid evaluation artifacts.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.experiment_log import append_run_log

ROOT = Path(__file__).resolve().parents[2]

PAIR_FEATURES_PATH = ROOT / "outputs" / "baselines" / "gold_B" / "pair_features.pkl"
RANKINGS_PATH = ROOT / "outputs" / "baselines" / "gold_B_ranking" / "rankings.pkl"

OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_hybrid"
FEATURES_PATH = OUTPUT_DIR / "pair_features_hybrid.pkl"
OOF_PATH = OUTPUT_DIR / "oof_predictions.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.pkl"
MODEL_PATH = OUTPUT_DIR / "logreg_model.pkl"

RANDOM_STATE = 0
N_SPLITS = 5
RANKING_TEXT_MODE = "concise"

BASE_FEATURE_COLUMNS = [
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

RANKING_FEATURE_COLUMNS = [
    "score_union",
    "rank_union",
    "inv_rank_union",
    "in_top1_union",
    "in_top2_union",
    "score_ja_preferred",
    "rank_ja_preferred",
    "inv_rank_ja_preferred",
    "in_top1_ja_preferred",
    "in_top2_ja_preferred",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + RANKING_FEATURE_COLUMNS


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


def record_level_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
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


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select the threshold that maximizes F1."""
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


def build_ranking_feature_table(rankings: pd.DataFrame) -> pd.DataFrame:
    """Pivot ranking outputs into pairwise features."""
    parts = []

    for mode in ["union", "ja_preferred"]:
        df_mode = rankings[rankings["candidate_mode"] == mode].copy()
        df_mode = df_mode.rename(
            columns={
                "score": f"score_{mode}",
                "rank": f"rank_{mode}",
            }
        )
        df_mode[f"inv_rank_{mode}"] = 1.0 / df_mode[f"rank_{mode}"]
        df_mode[f"in_top1_{mode}"] = (df_mode[f"rank_{mode}"] <= 1).astype(int)
        df_mode[f"in_top2_{mode}"] = (df_mode[f"rank_{mode}"] <= 2).astype(int)
        keep = [
            "record_id",
            "synset_id",
            f"score_{mode}",
            f"rank_{mode}",
            f"inv_rank_{mode}",
            f"in_top1_{mode}",
            f"in_top2_{mode}",
        ]
        parts.append(df_mode[keep])

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["record_id", "synset_id"], how="outer")

    out["record_id"] = out["record_id"].astype(int)
    out["synset_id"] = out["synset_id"].astype(str)
    return out


def build_hybrid_features(pair_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pairwise features and ranking features."""
    if "text_mode" in ranking_df.columns:
        ranking_df = ranking_df[ranking_df["text_mode"] == RANKING_TEXT_MODE].copy()

    ranking_features = build_ranking_feature_table(ranking_df)
    df = pair_df.merge(ranking_features, on=["record_id", "synset_id"], how="left")

    fill_values = {
        "score_union": -1.0,
        "rank_union": 999.0,
        "inv_rank_union": 0.0,
        "in_top1_union": 0,
        "in_top2_union": 0,
        "score_ja_preferred": -1.0,
        "rank_ja_preferred": 999.0,
        "inv_rank_ja_preferred": 0.0,
        "in_top1_ja_preferred": 0,
        "in_top2_ja_preferred": 0,
    }
    df = df.fillna(fill_values)
    return df


def run_grouped_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped cross-validation for the hybrid logistic regression."""
    groups = df["record_id"].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    gkf = GroupKFold(n_splits=N_SPLITS)
    out = df[["record_id", "synset_id", "label", "raw_label"]].copy()
    out["fold"] = -1
    out["hybrid_score"] = np.nan
    out["hybrid_pred"] = 0
    out["hybrid_threshold"] = np.nan

    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
        model.fit(X_train, y_train)

        train_score = model.predict_proba(X_train)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]
        threshold = best_threshold(y_train, train_score)
        test_pred = (test_score >= threshold).astype(int)

        out.loc[out.index[test_idx], "fold"] = fold_id
        out.loc[out.index[test_idx], "hybrid_score"] = test_score
        out.loc[out.index[test_idx], "hybrid_pred"] = test_pred
        out.loc[out.index[test_idx], "hybrid_threshold"] = threshold

        metrics = binary_metrics(y_test, test_pred)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train_pairs": len(train_idx),
                "n_test_pairs": len(test_idx),
                "hybrid_threshold": threshold,
                "hybrid_precision": metrics["precision"],
                "hybrid_recall": metrics["recall"],
                "hybrid_f1": metrics["f1"],
            }
        )

    return out, pd.DataFrame(fold_rows)


def summarize_results(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact hybrid metrics table."""
    pair_metrics = binary_metrics(
        oof["label"].to_numpy(dtype=int),
        oof["hybrid_pred"].to_numpy(dtype=int),
    )
    record_metrics = record_level_metrics(oof, "hybrid_pred")

    row = {
        "method": "hybrid_logreg",
        "pair_precision": pair_metrics["precision"],
        "pair_recall": pair_metrics["recall"],
        "pair_f1": pair_metrics["f1"],
        "pair_accuracy": pair_metrics["accuracy"],
        "record_macro_precision": record_metrics["record_macro_precision"],
        "record_macro_recall": record_metrics["record_macro_recall"],
        "record_macro_f1": record_metrics["record_macro_f1"],
        "record_exact_match_rate": record_metrics["record_exact_match_rate"],
        "mean_fold_f1": float(fold_metrics["hybrid_f1"].mean()),
        "std_fold_f1": float(fold_metrics["hybrid_f1"].std(ddof=0)),
    }
    return pd.DataFrame([row])


def train_final_model(df: pd.DataFrame) -> dict[str, object]:
    """Train the final hybrid logistic-regression model."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(X, y)

    full_score = model.predict_proba(X)[:, 1]
    threshold = best_threshold(y, full_score)

    return {
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "model": model,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_pickle(PAIR_FEATURES_PATH)
    ranking_df = pd.read_pickle(RANKINGS_PATH)

    hybrid_df = build_hybrid_features(pair_df, ranking_df)
    hybrid_df.to_pickle(FEATURES_PATH)
    print(f"Saved features: {FEATURES_PATH}")

    oof_df, fold_metrics = run_grouped_cv(hybrid_df)
    oof_df.to_pickle(OOF_PATH)
    print(f"Saved OOF predictions: {OOF_PATH}")

    metrics_df = summarize_results(oof_df, fold_metrics)
    metrics_obj = {
        "summary": metrics_df,
        "folds": fold_metrics,
    }
    with METRICS_PATH.open("wb") as f:
        pickle.dump(metrics_obj, f)
    print(f"Saved metrics: {METRICS_PATH}")

    final_model = train_final_model(hybrid_df)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(final_model, f)
    print(f"Saved final model: {MODEL_PATH}")

    print(metrics_df.to_string(index=False))

    summary_row = metrics_df.iloc[0].to_dict()
    append_run_log(
        run_name="gold_B_hybrid_logreg",
        rationale="pairwise baseline と ranking baseline を結合した hybrid を再実行",
        script_path="src/baselines/run_gold_b_hybrid.py",
        params={
            "ranking_text_mode": RANKING_TEXT_MODE,
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "ranking_feature_columns": ",".join(RANKING_FEATURE_COLUMNS),
        },
        metrics={
            "pair_f1": summary_row["pair_f1"],
            "pair_precision": summary_row["pair_precision"],
            "pair_recall": summary_row["pair_recall"],
            "record_macro_f1": summary_row["record_macro_f1"],
            "record_exact_match_rate": summary_row["record_exact_match_rate"],
        },
        outputs=[
            str(FEATURES_PATH.relative_to(ROOT)),
            str(OOF_PATH.relative_to(ROOT)),
            str(METRICS_PATH.relative_to(ROOT)),
            str(MODEL_PATH.relative_to(ROOT)),
        ],
    )


if __name__ == "__main__":
    main()
