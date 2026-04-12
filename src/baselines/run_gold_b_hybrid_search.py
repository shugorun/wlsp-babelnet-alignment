#!/usr/bin/env python
# Usage: python src/baselines/run_gold_b_hybrid_search.py --pair-features outputs/baselines/gold_B/pair_features.pkl --rankings outputs/baselines/gold_B_ranking/rankings.pkl --output-dir outputs/baselines/gold_B_hybrid_search --best-output-dir outputs/baselines/gold_B_hybrid_best
"""Search reproducible gold_B hybrid settings.

This script combines the pairwise baseline features with ranking outputs,
evaluates several hybrid feature presets, and exports the best reproducible
configuration under outputs/baselines/gold_B_hybrid_best.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
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

DEFAULT_PAIR_FEATURES_PATH = ROOT / "outputs" / "baselines" / "gold_B" / "pair_features.pkl"
DEFAULT_RANKINGS_PATH = ROOT / "outputs" / "baselines" / "gold_B_ranking" / "rankings.pkl"

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_hybrid_search"
DEFAULT_BEST_OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_hybrid_best"

RANDOM_STATE = 0
N_SPLITS = 5

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

FEATURE_PRESETS = {
    "full_both_top2": [
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
    ],
    "ja_preferred_top1": [
        "score_ja_preferred",
        "rank_ja_preferred",
        "inv_rank_ja_preferred",
        "in_top1_ja_preferred",
    ],
    "ja_preferred_top2": [
        "score_ja_preferred",
        "rank_ja_preferred",
        "inv_rank_ja_preferred",
        "in_top1_ja_preferred",
        "in_top2_ja_preferred",
    ],
    "compact_both_top1": [
        "score_union",
        "rank_union",
        "inv_rank_union",
        "in_top1_union",
        "score_ja_preferred",
        "rank_ja_preferred",
        "inv_rank_ja_preferred",
        "in_top1_ja_preferred",
    ],
}

# Each preset adds a different subset of ranking features on top of the
# lexical pairwise baseline. The search keeps this comparison explicit.


def path_for_log(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

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
    return df.fillna(fill_values)


def run_grouped_cv(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped cross-validation for one hybrid feature set."""
    groups = df["record_id"].to_numpy()
    X = df[feature_columns].to_numpy(dtype=float)
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

        pair_metrics = binary_metrics(y_test, test_pred)
        fold_rows.append(
            {
                "fold": fold_id,
                "hybrid_threshold": threshold,
                "hybrid_precision": pair_metrics["precision"],
                "hybrid_recall": pair_metrics["recall"],
                "hybrid_f1": pair_metrics["f1"],
            }
        )

    return out, pd.DataFrame(fold_rows)


def summarize_results(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> dict[str, float]:
    """Return a compact metrics summary for one hybrid run."""
    pair_metrics = binary_metrics(
        oof["label"].to_numpy(dtype=int),
        oof["hybrid_pred"].to_numpy(dtype=int),
    )
    record_metrics = record_level_metrics(oof, "hybrid_pred")

    return {
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


def train_final_model(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    """Train the final hybrid model for one selected feature set."""
    X = df[feature_columns].to_numpy(dtype=float)
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
        "feature_columns": feature_columns,
        "threshold": threshold,
        "model": model,
    }


def search_best_config(pair_df: pd.DataFrame, rankings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Search text mode and feature preset combinations."""
    rows = []
    best = None

    # Search is intentionally small and reproducible: ranking template x
    # feature preset, scored with grouped cross-validation.
    for text_mode in sorted(rankings["text_mode"].unique()):
        ranking_mode_df = rankings[rankings["text_mode"] == text_mode].copy()
        hybrid_df = build_hybrid_features(pair_df, ranking_mode_df)

        query_mode = ranking_mode_df["query_mode"].iloc[0]
        passage_mode = ranking_mode_df["passage_mode"].iloc[0]

        for preset_name, ranking_feature_columns in FEATURE_PRESETS.items():
            feature_columns = BASE_FEATURE_COLUMNS + ranking_feature_columns
            oof_df, fold_df = run_grouped_cv(hybrid_df, feature_columns)
            metrics = summarize_results(oof_df, fold_df)

            row = {
                "text_mode": text_mode,
                "query_mode": query_mode,
                "passage_mode": passage_mode,
                "feature_preset": preset_name,
                "feature_columns": feature_columns,
                "hybrid_df": hybrid_df,
                "oof_df": oof_df,
                "fold_df": fold_df,
                **metrics,
            }
            rows.append(row)

            current = (
                metrics["pair_f1"],
                metrics["record_macro_f1"],
                metrics["record_exact_match_rate"],
            )
            if best is None:
                best = row
                continue

            previous = (
                best["pair_f1"],
                best["record_macro_f1"],
                best["record_exact_match_rate"],
            )
            if current > previous:
                best = row

    summary_df = pd.DataFrame(
        [
            {
                "text_mode": row["text_mode"],
                "query_mode": row["query_mode"],
                "passage_mode": row["passage_mode"],
                "feature_preset": row["feature_preset"],
                "pair_precision": row["pair_precision"],
                "pair_recall": row["pair_recall"],
                "pair_f1": row["pair_f1"],
                "pair_accuracy": row["pair_accuracy"],
                "record_macro_f1": row["record_macro_f1"],
                "record_exact_match_rate": row["record_exact_match_rate"],
            }
            for row in rows
        ]
    ).sort_values(
        ["pair_f1", "record_macro_f1", "record_exact_match_rate"],
        ascending=False,
    )

    return summary_df, best


def export_best(best: dict[str, object], best_output_dir: Path) -> dict[str, Path]:
    """Persist the best reproducible hybrid configuration."""
    best_output_dir.mkdir(parents=True, exist_ok=True)
    best_features_path = best_output_dir / "pair_features_hybrid.pkl"
    best_oof_path = best_output_dir / "oof_predictions.pkl"
    best_metrics_path = best_output_dir / "metrics.pkl"
    best_model_path = best_output_dir / "logreg_model.pkl"
    best_config_path = best_output_dir / "config.json"

    hybrid_df = best["hybrid_df"]
    oof_df = best["oof_df"]
    fold_df = best["fold_df"]
    feature_columns = best["feature_columns"]
    final_model = train_final_model(hybrid_df, feature_columns)

    hybrid_df.to_pickle(best_features_path)
    oof_df.to_pickle(best_oof_path)

    metrics_df = pd.DataFrame(
        [
            {
                "method": "hybrid_logreg_best",
                "pair_precision": best["pair_precision"],
                "pair_recall": best["pair_recall"],
                "pair_f1": best["pair_f1"],
                "pair_accuracy": best["pair_accuracy"],
                "record_macro_precision": best["record_macro_precision"],
                "record_macro_recall": best["record_macro_recall"],
                "record_macro_f1": best["record_macro_f1"],
                "record_exact_match_rate": best["record_exact_match_rate"],
                "mean_fold_f1": best["mean_fold_f1"],
                "std_fold_f1": best["std_fold_f1"],
            }
        ]
    )
    with best_metrics_path.open("wb") as f:
        pickle.dump({"summary": metrics_df, "folds": fold_df}, f)

    with best_model_path.open("wb") as f:
        pickle.dump(final_model, f)

    # Save a lightweight config alongside the full model bundle so the same
    # setting can be applied later to gold_A.
    config = {
        "config_name": "gold_b_hybrid_best_reconstructed",
        "text_mode": best["text_mode"],
        "query_mode": best["query_mode"],
        "passage_mode": best["passage_mode"],
        "feature_preset": best["feature_preset"],
        "feature_columns": feature_columns,
        "threshold": final_model["threshold"],
    }
    best_config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "features": best_features_path,
        "oof": best_oof_path,
        "metrics": best_metrics_path,
        "model": best_model_path,
        "config": best_config_path,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-features", type=Path, default=DEFAULT_PAIR_FEATURES_PATH)
    ap.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--best-output-dir", type=Path, default=DEFAULT_BEST_OUTPUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    summary_path = output_dir / "summary.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)

    # The hybrid stage reuses exported artifacts from the lexical and ranking
    # baselines instead of rebuilding them here.
    pair_df = pd.read_pickle(args.pair_features)
    rankings = pd.read_pickle(args.rankings)

    summary_df, best = search_best_config(pair_df, rankings)
    with summary_path.open("wb") as f:
        pickle.dump({"summary": summary_df}, f)
    print(summary_df.head(20).to_string(index=False))

    best_paths = export_best(best, args.best_output_dir)

    append_run_log(
        run_name="gold_B_hybrid_search",
        rationale="Search reproducible hybrid settings by combining the ranking templates with several feature presets, then export the best configuration.",
        script_path="src/baselines/run_gold_b_hybrid_search.py",
        params={
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "n_text_modes": int(rankings["text_mode"].nunique()),
            "feature_presets": ",".join(FEATURE_PRESETS.keys()),
        },
        metrics={
            "best_text_mode": best["text_mode"],
            "best_feature_preset": best["feature_preset"],
            "best_pair_f1": best["pair_f1"],
            "best_pair_precision": best["pair_precision"],
            "best_pair_recall": best["pair_recall"],
            "best_record_exact_match_rate": best["record_exact_match_rate"],
        },
        outputs=[
            path_for_log(summary_path),
            path_for_log(best_paths["features"]),
            path_for_log(best_paths["oof"]),
            path_for_log(best_paths["metrics"]),
            path_for_log(best_paths["model"]),
            path_for_log(best_paths["config"]),
        ],
    )


if __name__ == "__main__":
    main()
