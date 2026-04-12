#!/usr/bin/env python
# Usage: python src/alignment/evaluate_alignment.py --gold data/gold/gold_A.pkl --pred-dir outputs/api_runs/alignment/gpt-5.4-2026-03-05_high_v1_v1/gold_A/parsed --out-dir outputs/api_runs/alignment/gpt-5.4-2026-03-05_high_v1_v1/gold_A/evaluation
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GOLD_PATH = ROOT / "data" / "gold" / "gold_B.pkl"
DEFAULT_PRED_DIR = ROOT / "outputs" / "api_runs" / "alignment" / "gpt-5.2-2025-12-11_high_v1_v1" / "gold_B" / "parsed"
DEFAULT_OUT_DIR = ROOT / "outputs" / "api_runs" / "alignment" / "gpt-5.2-2025-12-11_high_v1_v1" / "gold_B" / "evaluation"

LABELS = ["EQUAL", "HYPERNYM", "HYPONYM", "NONE"]


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    for path in sorted(pred_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        record_id = int(obj["record_id"])

        for item in obj.get("labels", []):
            rows.append(
                {
                    "record_id": record_id,
                    "synset_id": str(item["synset_id"]),
                    "pred_label": str(item["label"]),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No parsed prediction files were found in: {pred_dir}")

    pred = pd.DataFrame(rows)
    pred = pred.drop_duplicates(subset=["record_id", "synset_id"], keep="last").copy()
    return pred


def per_label_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label in LABELS:
        tp = int(((df["gold_label"] == label) & (df["pred_label"] == label)).sum())
        fp = int(((df["gold_label"] != label) & (df["pred_label"] == label)).sum())
        fn = int(((df["gold_label"] == label) & (df["pred_label"] != label)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int((df["gold_label"] == label).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return pd.DataFrame(rows)


def equal_vs_rest_metrics(df: pd.DataFrame) -> Dict[str, float]:
    gold_pos = df["gold_label"] == "EQUAL"
    pred_pos = df["pred_label"] == "EQUAL"

    tp = int((gold_pos & pred_pos).sum())
    fp = int((~gold_pos & pred_pos).sum())
    fn = int((gold_pos & ~pred_pos).sum())
    tn = int((~gold_pos & ~pred_pos).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(df) if len(df) else 0.0

    return {
        "precision_equal": precision,
        "recall_equal": recall,
        "f1_equal": f1,
        "accuracy_equal_vs_rest": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def record_exact_match_rate(df: pd.DataFrame) -> float:
    exact = []

    for _, group in df.groupby("record_id", sort=False):
        gold_pairs = {(sid, label) for sid, label in zip(group["synset_id"], group["gold_label"])}
        pred_pairs = {(sid, label) for sid, label in zip(group["synset_id"], group["pred_label"])}
        exact.append(float(gold_pairs == pred_pairs))

    return sum(exact) / len(exact) if exact else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    ap.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = pd.read_pickle(args.gold).copy()
    gold["record_id"] = gold["record_id"].astype(int)
    gold["synset_id"] = gold["synset_id"].astype(str)
    gold["gold_label"] = gold["label"].astype(str)
    gold = gold[["record_id", "synset_id", "gold_label"]]

    pred = load_predictions(args.pred_dir)

    merged = gold.merge(pred, on=["record_id", "synset_id"], how="left")
    merged["pred_label"] = merged["pred_label"].fillna("MISSING")

    missing = merged["pred_label"].eq("MISSING").sum()
    if missing:
        raise RuntimeError(f"Missing predictions for {missing} gold pairs.")

    pair_accuracy = float((merged["gold_label"] == merged["pred_label"]).mean())
    confusion = pd.crosstab(
        merged["gold_label"],
        merged["pred_label"],
        rownames=["gold_label"],
        colnames=["pred_label"],
        dropna=False,
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    per_class = per_label_metrics(merged)
    equal_rest = equal_vs_rest_metrics(merged)
    exact_match = record_exact_match_rate(merged)

    summary = {
        "n_pairs": int(len(merged)),
        "n_records": int(merged["record_id"].nunique()),
        "pair_accuracy_multiclass": pair_accuracy,
        "record_exact_match_rate": exact_match,
        **equal_rest,
    }

    merged.to_csv(args.out_dir / "pair_predictions.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(args.out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    per_class.to_csv(args.out_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(per_class.to_string(index=False))
    print()
    print(confusion.to_string())


if __name__ == "__main__":
    main()
