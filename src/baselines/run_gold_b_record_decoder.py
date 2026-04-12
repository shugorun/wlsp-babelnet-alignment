#!/usr/bin/env python
# Usage: python src/baselines/run_gold_b_record_decoder.py --hybrid-features outputs/baselines/gold_B_hybrid_best/pair_features_hybrid.pkl --output-dir outputs/baselines/gold_B_record_decoder
"""Decode hybrid pairwise scores at the record level on gold_B.

This script trains the hybrid scorer under grouped cross-validation and
searches simple record-level decoding rules on each training split. The
selected rule is then applied to the held-out records and evaluated with
record-level set metrics.
"""

from __future__ import annotations

import argparse
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

DEFAULT_HYBRID_FEATURES_PATH = ROOT / "outputs" / "baselines" / "gold_B_hybrid_best" / "pair_features_hybrid.pkl"

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_record_decoder"

RANDOM_STATE = 0
N_SPLITS = 5

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

PARAM_GRID = {
    "max_k": [1, 2, 3],
    "score_min": [0.50, 0.60, 0.70, 0.80, 0.85],
    "ratio_min": [0.60, 0.70, 0.80, 0.90, 0.95],
}

# The decoder controls how many synsets to return per record after the
# hybrid scorer has produced pairwise probabilities.


def path_for_log(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

def build_model():
    """Build the hybrid logistic-regression scorer."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def decode_records(df: pd.DataFrame, max_k: int, score_min: float, ratio_min: float) -> pd.DataFrame:
    """Decode pairwise scores into record-level predictions."""
    out = df.copy()
    out["decoded_pred"] = 0

    sort_cols = ["hybrid_score"]
    if "score_ja_preferred" in out.columns:
        sort_cols.append("score_ja_preferred")
    if "score_union" in out.columns:
        sort_cols.append("score_union")

    # Keep the top candidate and optionally add near-ties when they are both
    # strong enough and close to the top score.
    for _, group in out.groupby("record_id", sort=False):
        ranked = group.sort_values(sort_cols, ascending=False)
        if ranked.empty:
            continue

        top1_score = float(ranked.iloc[0]["hybrid_score"])
        keep_index = [ranked.index[0]]

        for idx, row in ranked.iloc[1:max_k].iterrows():
            score = float(row["hybrid_score"])
            ratio = score / top1_score if top1_score > 0 else 0.0
            if score >= score_min and ratio >= ratio_min:
                keep_index.append(idx)

        out.loc[keep_index, "decoded_pred"] = 1

    return out


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


def pair_level_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Return pair-level EQUAL-vs-rest metrics."""
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = df[pred_col].to_numpy(dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0

    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "pair_accuracy": accuracy,
    }


def search_decoder_params(train_df: pd.DataFrame) -> dict[str, float]:
    """Search decoder parameters with exact match as the main objective."""
    best = None
    best_metrics = None

    for max_k in PARAM_GRID["max_k"]:
        for score_min in PARAM_GRID["score_min"]:
            for ratio_min in PARAM_GRID["ratio_min"]:
                decoded = decode_records(train_df, max_k=max_k, score_min=score_min, ratio_min=ratio_min)
                metrics = record_level_metrics(decoded, "decoded_pred")

                if best is None:
                    best = {"max_k": max_k, "score_min": score_min, "ratio_min": ratio_min}
                    best_metrics = metrics
                    continue

                current = (
                    metrics["record_exact_match_rate"],
                    metrics["record_macro_f1"],
                    metrics["record_macro_precision"],
                )
                previous = (
                    best_metrics["record_exact_match_rate"],
                    best_metrics["record_macro_f1"],
                    best_metrics["record_macro_precision"],
                )
                if current > previous:
                    best = {"max_k": max_k, "score_min": score_min, "ratio_min": ratio_min}
                    best_metrics = metrics

    return best


def run_grouped_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped CV for the hybrid scorer and record decoder."""
    groups = df["record_id"].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    gkf = GroupKFold(n_splits=N_SPLITS)
    out = df[["record_id", "synset_id", "label", "raw_label"]].copy()
    out["fold"] = -1
    out["hybrid_score"] = np.nan
    out["decoded_pred"] = 0

    fold_rows = []

    # Learn the pairwise scorer on the training records, tune decoder
    # parameters on the same training split, and apply them to held-out records.
    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]

        model = build_model()
        model.fit(X_train, y_train)

        train_df = df.iloc[train_idx][
            ["record_id", "synset_id", "label", "raw_label", "score_ja_preferred", "score_union"]
        ].copy()
        train_df["hybrid_score"] = model.predict_proba(X_train)[:, 1]

        test_df = df.iloc[test_idx][
            ["record_id", "synset_id", "label", "raw_label", "score_ja_preferred", "score_union"]
        ].copy()
        test_df["hybrid_score"] = model.predict_proba(X_test)[:, 1]

        params = search_decoder_params(train_df)
        decoded_test = decode_records(test_df, **params)

        out.loc[out.index[test_idx], "fold"] = fold_id
        out.loc[out.index[test_idx], "hybrid_score"] = decoded_test["hybrid_score"].to_numpy()
        out.loc[out.index[test_idx], "decoded_pred"] = decoded_test["decoded_pred"].to_numpy()

        record_metrics = record_level_metrics(decoded_test, "decoded_pred")
        pair_metrics = pair_level_metrics(decoded_test, "decoded_pred")
        fold_rows.append(
            {
                "fold": fold_id,
                "max_k": params["max_k"],
                "score_min": params["score_min"],
                "ratio_min": params["ratio_min"],
                "record_macro_f1": record_metrics["record_macro_f1"],
                "record_exact_match_rate": record_metrics["record_exact_match_rate"],
                "pair_f1": pair_metrics["pair_f1"],
            }
        )

    return out, pd.DataFrame(fold_rows)


def summarize_results(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary table."""
    record_metrics = record_level_metrics(oof, "decoded_pred")
    pair_metrics = pair_level_metrics(oof, "decoded_pred")

    row = {
        "method": "hybrid_record_decoder",
        "pair_precision": pair_metrics["pair_precision"],
        "pair_recall": pair_metrics["pair_recall"],
        "pair_f1": pair_metrics["pair_f1"],
        "pair_accuracy": pair_metrics["pair_accuracy"],
        "record_macro_precision": record_metrics["record_macro_precision"],
        "record_macro_recall": record_metrics["record_macro_recall"],
        "record_macro_f1": record_metrics["record_macro_f1"],
        "record_exact_match_rate": record_metrics["record_exact_match_rate"],
        "mean_fold_record_macro_f1": float(fold_metrics["record_macro_f1"].mean()),
        "std_fold_record_macro_f1": float(fold_metrics["record_macro_f1"].std(ddof=0)),
    }
    return pd.DataFrame([row])


def train_final_model(df: pd.DataFrame) -> dict[str, object]:
    """Train the final scorer and select decoder params on all gold_B pairs."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    model = build_model()
    model.fit(X, y)

    scored = df[["record_id", "synset_id", "label", "raw_label", "score_ja_preferred", "score_union"]].copy()
    scored["hybrid_score"] = model.predict_proba(X)[:, 1]
    params = search_decoder_params(scored)

    return {
        "feature_columns": FEATURE_COLUMNS,
        "decoder_params": params,
        "model": model,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-features", type=Path, default=DEFAULT_HYBRID_FEATURES_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    oof_path = output_dir / "oof_predictions.pkl"
    metrics_path = output_dir / "metrics.pkl"
    model_path = output_dir / "logreg_model.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.hybrid_features)

    oof_df, fold_metrics = run_grouped_cv(df)
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

    final_model = train_final_model(df)
    with model_path.open("wb") as f:
        pickle.dump(final_model, f)
    print(f"Saved final model: {model_path}")

    print(metrics_df.to_string(index=False))

    summary_row = metrics_df.iloc[0].to_dict()
    append_run_log(
        run_name="gold_B_record_decoder",
        rationale="Decode the best hybrid pairwise scores at the record level and optimize for exact-match oriented output control.",
        script_path="src/baselines/run_gold_b_record_decoder.py",
        params={
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "param_grid_max_k": ",".join(str(x) for x in PARAM_GRID["max_k"]),
            "param_grid_score_min": ",".join(str(x) for x in PARAM_GRID["score_min"]),
            "param_grid_ratio_min": ",".join(str(x) for x in PARAM_GRID["ratio_min"]),
        },
        metrics={
            "pair_f1": summary_row["pair_f1"],
            "record_macro_f1": summary_row["record_macro_f1"],
            "record_exact_match_rate": summary_row["record_exact_match_rate"],
        },
        outputs=[
            path_for_log(oof_path),
            path_for_log(metrics_path),
            path_for_log(model_path),
        ],
    )


if __name__ == "__main__":
    main()
