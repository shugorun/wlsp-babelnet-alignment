"""Evaluate a pretrained cross-encoder on gold_B pair candidates.

This script uses a pretrained reranker as-is and does not fine-tune it.
It scores each record_id x synset_id pair, tunes a threshold on train folds,
and evaluates with grouped cross-validation on gold_B.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.run_gold_b_ranking import build_passage_text, build_query_text

ROOT = Path(__file__).resolve().parents[2]

RECORDS_PATH = ROOT / "data" / "gold" / "gold_B_records.pkl"
GOLD_PATH = ROOT / "data" / "gold" / "gold_B.pkl"
BABELNET_PATH = ROOT / "data" / "processed" / "babelnet_.pkl"

OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_cross_encoder"
PAIR_TEXTS_PATH = OUTPUT_DIR / "pair_texts.pkl"
OOF_PATH = OUTPUT_DIR / "oof_predictions.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.pkl"

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
TEXT_MODE = "concise"
MAX_LENGTH = 512
N_SPLITS = 5
RANDOM_STATE = 0
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    best_t = float(candidates[0]) if len(candidates) else 0.5
    best_f1 = -1.0

    for threshold in candidates:
        y_pred = scores >= threshold
        metrics = binary_metrics(y_true, y_pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(threshold)

    return best_t


def build_pair_table(records: pd.DataFrame, gold: pd.DataFrame, babelnet: pd.DataFrame) -> pd.DataFrame:
    """Build one text pair per gold_B candidate pair."""
    rows = []

    for pair in gold.itertuples(index=False):
        rid = int(pair.record_id)
        sid = str(pair.synset_id)

        record = records.loc[rid]
        synset = babelnet.loc[sid]

        rows.append(
            {
                "record_id": rid,
                "synset_id": sid,
                "label": 1 if str(pair.label) == "EQUAL" else 0,
                "raw_label": str(pair.label),
                "query_text": build_query_text(record, TEXT_MODE),
                "passage_text": build_passage_text(synset, TEXT_MODE),
            }
        )

    return pd.DataFrame(rows)


def load_model():
    """Load the pretrained cross-encoder."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model


def score_pairs(tokenizer, model, pair_df: pd.DataFrame, desc: str) -> np.ndarray:
    """Score query/passage pairs with a pretrained reranker."""
    scores = []
    total = len(pair_df)

    for start in tqdm(range(0, total, BATCH_SIZE), desc=desc, unit="batch"):
        batch = pair_df.iloc[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch["query_text"].tolist(),
            batch["passage_text"].tolist(),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits

        batch_scores = logits.view(-1).detach().cpu().numpy()
        scores.append(batch_scores)

    if not scores:
        return np.array([], dtype=float)

    return np.concatenate(scores, axis=0)


def run_grouped_cv(pair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped cross-validation with one pretrained model."""
    groups = pair_df["record_id"].to_numpy()
    y = pair_df["label"].to_numpy(dtype=int)

    tokenizer, model = load_model()

    gkf = GroupKFold(n_splits=N_SPLITS)
    out = pair_df[["record_id", "synset_id", "label", "raw_label"]].copy()
    out["fold"] = -1
    out["cross_score"] = np.nan
    out["cross_pred"] = 0
    out["cross_threshold"] = np.nan

    fold_rows = []

    split_iter = gkf.split(pair_df, y, groups)
    for fold_id, (train_idx, test_idx) in enumerate(
        tqdm(split_iter, total=N_SPLITS, desc="cross-validation", unit="fold"),
        start=1,
    ):
        print(f"[INFO] fold {fold_id}/{N_SPLITS}: train={len(train_idx)} test={len(test_idx)}")

        train_df = pair_df.iloc[train_idx].reset_index(drop=True)
        test_df = pair_df.iloc[test_idx].reset_index(drop=True)

        train_scores = score_pairs(tokenizer, model, train_df, f"score train fold {fold_id}")
        test_scores = score_pairs(tokenizer, model, test_df, f"score test fold {fold_id}")
        threshold = best_threshold(train_df["label"].to_numpy(dtype=int), train_scores)
        test_pred = (test_scores >= threshold).astype(int)

        out.loc[out.index[test_idx], "fold"] = fold_id
        out.loc[out.index[test_idx], "cross_score"] = test_scores
        out.loc[out.index[test_idx], "cross_pred"] = test_pred
        out.loc[out.index[test_idx], "cross_threshold"] = threshold

        metrics = binary_metrics(test_df["label"].to_numpy(dtype=int), test_pred)
        fold_rows.append(
            {
                "fold": fold_id,
                "pair_precision": metrics["precision"],
                "pair_recall": metrics["recall"],
                "pair_f1": metrics["f1"],
                "threshold": threshold,
            }
        )
        print(
            f"[INFO] fold {fold_id} done: "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"threshold={threshold:.4f}"
        )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out, pd.DataFrame(fold_rows)


def summarize_results(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact metrics table."""
    pair_metrics = binary_metrics(
        oof["label"].to_numpy(dtype=int),
        oof["cross_pred"].to_numpy(dtype=int),
    )
    record_metrics = record_level_metrics(oof, "cross_pred")

    row = {
        "method": "cross_encoder_pretrained",
        "model_name": MODEL_NAME,
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


def main() -> None:
    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = pd.read_pickle(RECORDS_PATH)
    gold = pd.read_pickle(GOLD_PATH)
    babelnet = pd.read_pickle(BABELNET_PATH)
    babelnet.index = babelnet.index.map(str)

    pair_df = build_pair_table(records, gold, babelnet)
    pair_df.to_pickle(PAIR_TEXTS_PATH)
    print(f"Saved pair texts: {PAIR_TEXTS_PATH}")

    oof_df, fold_metrics = run_grouped_cv(pair_df)
    oof_df.to_pickle(OOF_PATH)
    print(f"Saved OOF predictions: {OOF_PATH}")

    metrics_df = summarize_results(oof_df, fold_metrics)
    with METRICS_PATH.open("wb") as f:
        pickle.dump({"summary": metrics_df, "folds": fold_metrics}, f)
    print(f"Saved metrics: {METRICS_PATH}")

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
