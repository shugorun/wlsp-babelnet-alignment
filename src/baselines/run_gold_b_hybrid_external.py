"""Combine pairwise features with E5 and MPNet ranking features on gold_B."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

PAIR_FEATURES_PATH = ROOT / "outputs" / "baselines" / "gold_B" / "pair_features.pkl"
E5_RANKINGS_PATH = ROOT / "outputs" / "baselines" / "gold_B_ranking" / "rankings.pkl"
MPNET_RANKINGS_PATH = ROOT / "outputs" / "baselines" / "gold_B_ranking_mpnet" / "rankings.pkl"

OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_hybrid_external"
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


def make_ranking_feature_columns(prefix: str) -> list[str]:
    """Return ranking feature names for one model prefix."""
    return [
        f"score_union_{prefix}",
        f"rank_union_{prefix}",
        f"inv_rank_union_{prefix}",
        f"in_top1_union_{prefix}",
        f"in_top2_union_{prefix}",
        f"score_ja_preferred_{prefix}",
        f"rank_ja_preferred_{prefix}",
        f"inv_rank_ja_preferred_{prefix}",
        f"in_top1_ja_preferred_{prefix}",
        f"in_top2_ja_preferred_{prefix}",
    ]


RANKING_FEATURE_COLUMNS = make_ranking_feature_columns("e5") + make_ranking_feature_columns("mpnet")
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
    best_t = float(candidates[0]) if len(candidates) else 0.0
    best_f1 = -1.0

    for threshold in candidates:
        y_pred = scores >= threshold
        metrics = binary_metrics(y_true, y_pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(threshold)

    return best_t


def build_ranking_feature_table(rankings: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Pivot one ranking output into pairwise features."""
    if "text_mode" in rankings.columns:
        rankings = rankings[rankings["text_mode"] == RANKING_TEXT_MODE].copy()

    parts = []
    for mode in ["union", "ja_preferred"]:
        df_mode = rankings[rankings["candidate_mode"] == mode].copy()
        df_mode = df_mode.rename(
            columns={
                "score": f"score_{mode}_{prefix}",
                "rank": f"rank_{mode}_{prefix}",
            }
        )
        df_mode[f"inv_rank_{mode}_{prefix}"] = 1.0 / df_mode[f"rank_{mode}_{prefix}"]
        df_mode[f"in_top1_{mode}_{prefix}"] = (df_mode[f"rank_{mode}_{prefix}"] <= 1).astype(int)
        df_mode[f"in_top2_{mode}_{prefix}"] = (df_mode[f"rank_{mode}_{prefix}"] <= 2).astype(int)
        keep = [
            "record_id",
            "synset_id",
            f"score_{mode}_{prefix}",
            f"rank_{mode}_{prefix}",
            f"inv_rank_{mode}_{prefix}",
            f"in_top1_{mode}_{prefix}",
            f"in_top2_{mode}_{prefix}",
        ]
        parts.append(df_mode[keep])

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["record_id", "synset_id"], how="outer")

    out["record_id"] = out["record_id"].astype(int)
    out["synset_id"] = out["synset_id"].astype(str)
    return out


def build_features(pair_df: pd.DataFrame, e5_rankings: pd.DataFrame, mpnet_rankings: pd.DataFrame) -> pd.DataFrame:
    """Merge pairwise features with both ranking feature tables."""
    e5_features = build_ranking_feature_table(e5_rankings, "e5")
    mpnet_features = build_ranking_feature_table(mpnet_rankings, "mpnet")

    df = pair_df.merge(e5_features, on=["record_id", "synset_id"], how="left")
    df = df.merge(mpnet_features, on=["record_id", "synset_id"], how="left")

    fill_values = {}
    for prefix in ["e5", "mpnet"]:
        fill_values.update(
            {
                f"score_union_{prefix}": -1.0,
                f"rank_union_{prefix}": 999.0,
                f"inv_rank_union_{prefix}": 0.0,
                f"in_top1_union_{prefix}": 0,
                f"in_top2_union_{prefix}": 0,
                f"score_ja_preferred_{prefix}": -1.0,
                f"rank_ja_preferred_{prefix}": 999.0,
                f"inv_rank_ja_preferred_{prefix}": 0.0,
                f"in_top1_ja_preferred_{prefix}": 0,
                f"in_top2_ja_preferred_{prefix}": 0,
            }
        )
    return df.fillna(fill_values)


def run_grouped_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped cross-validation for the external hybrid model."""
    groups = df["record_id"].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    gkf = GroupKFold(n_splits=N_SPLITS)
    out = df[["record_id", "synset_id", "label", "raw_label"]].copy()
    out["fold"] = -1
    out["hybrid_score"] = np.nan
    out["hybrid_pred"] = 0

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

        metrics = binary_metrics(y_test, test_pred)
        fold_rows.append(
            {
                "fold": fold_id,
                "pair_precision": metrics["precision"],
                "pair_recall": metrics["recall"],
                "pair_f1": metrics["f1"],
            }
        )

    return out, pd.DataFrame(fold_rows)


def summarize_results(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact metrics table."""
    pair_metrics = binary_metrics(
        oof["label"].to_numpy(dtype=int),
        oof["hybrid_pred"].to_numpy(dtype=int),
    )
    record_metrics = record_level_metrics(oof, "hybrid_pred")

    row = {
        "method": "hybrid_external",
        "pair_precision": pair_metrics["precision"],
        "pair_recall": pair_metrics["recall"],
        "pair_f1": pair_metrics["f1"],
        "pair_accuracy": pair_metrics["accuracy"],
        "record_macro_precision": record_metrics["record_macro_precision"],
        "record_macro_recall": record_metrics["record_macro_recall"],
        "record_macro_f1": record_metrics["record_macro_f1"],
        "record_exact_match_rate": record_metrics["record_exact_match_rate"],
        "mean_fold_f1": float(fold_metrics["pair_f1"].mean()),
        "std_fold_f1": float(fold_metrics["pair_f1"].std(ddof=0)),
    }
    return pd.DataFrame([row])


def train_final_model(df: pd.DataFrame) -> dict[str, object]:
    """Train the final external hybrid model."""
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

    threshold = best_threshold(y, model.predict_proba(X)[:, 1])
    return {
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "model": model,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_pickle(PAIR_FEATURES_PATH)
    e5_rankings = pd.read_pickle(E5_RANKINGS_PATH)
    mpnet_rankings = pd.read_pickle(MPNET_RANKINGS_PATH)

    feature_df = build_features(pair_df, e5_rankings, mpnet_rankings)
    feature_df.to_pickle(FEATURES_PATH)
    print(f"Saved features: {FEATURES_PATH}")

    oof_df, fold_metrics = run_grouped_cv(feature_df)
    oof_df.to_pickle(OOF_PATH)
    print(f"Saved OOF predictions: {OOF_PATH}")

    metrics_df = summarize_results(oof_df, fold_metrics)
    with METRICS_PATH.open("wb") as f:
        pickle.dump({"summary": metrics_df, "folds": fold_metrics}, f)
    print(f"Saved metrics: {METRICS_PATH}")

    final_model = train_final_model(feature_df)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(final_model, f)
    print(f"Saved final model: {MODEL_PATH}")

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
