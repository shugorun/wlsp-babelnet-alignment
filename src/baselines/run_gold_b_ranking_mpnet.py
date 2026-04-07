"""Rank gold_B candidates with a local Sentence-Transformers model.

This script uses the cached paraphrase-multilingual-mpnet-base-v2 model
from the local Hugging Face cache and evaluates candidate ranking on
gold_B without any API calls.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.run_gold_b_ranking import build_passage_text, build_query_text

ROOT = Path(__file__).resolve().parents[2]

RECORDS_PATH = ROOT / "data" / "gold" / "gold_B_records.pkl"
GOLD_PATH = ROOT / "data" / "gold" / "gold_B.pkl"
BABELNET_PATH = ROOT / "data" / "processed" / "babelnet_.pkl"

OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_ranking_mpnet"
RANKINGS_PATH = OUTPUT_DIR / "rankings.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.pkl"

TEXT_MODE = "concise"
TOPKS = [1, 2, 3, 5]
MODEL_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"
    / "snapshots"
    / "4328cf26390c98c5e3c738b4460a05b95f4911f5"
)


def dedupe_keep_order(items: list[str]) -> list[str]:
    """Return unique strings while keeping order."""
    return list(dict.fromkeys(items))


def build_rankings(records: pd.DataFrame, gold: pd.DataFrame, babelnet: pd.DataFrame) -> pd.DataFrame:
    """Build candidate rankings for union and JA-preferred settings."""
    model = SentenceTransformer(str(MODEL_DIR))

    record_ids = records.index.astype(int).tolist()
    query_texts = [build_query_text(records.loc[rid], TEXT_MODE) for rid in record_ids]
    query_vecs = model.encode(
        query_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vec_by_rid = {rid: query_vecs[i] for i, rid in enumerate(record_ids)}

    candidate_sids = sorted({str(sid) for sid in gold["synset_id"].astype(str).tolist()})
    passage_texts = [build_passage_text(babelnet.loc[sid], TEXT_MODE) for sid in candidate_sids]
    passage_vecs = model.encode(
        passage_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    passage_vec_by_sid = {sid: passage_vecs[i] for i, sid in enumerate(candidate_sids)}

    rows = []

    for rid, row in records.iterrows():
        rid = int(rid)
        query_vec = query_vec_by_rid[rid]

        sids_ja = dedupe_keep_order([str(sid) for sid in row.get("sids_JA", [])])
        sids_jm = dedupe_keep_order([str(sid) for sid in row.get("sids_JMdict", [])])

        candidate_specs = {
            "union": dedupe_keep_order(sids_ja + sids_jm),
            "ja_preferred": sids_ja if sids_ja else sids_jm,
        }

        for candidate_mode, candidate_sids_for_row in candidate_specs.items():
            filtered = [sid for sid in candidate_sids_for_row if sid in passage_vec_by_sid]
            if not filtered:
                continue

            scores = []
            for sid in filtered:
                score = float(np.dot(query_vec, passage_vec_by_sid[sid]))
                scores.append((sid, score))
            scores.sort(key=lambda item: item[1], reverse=True)

            for rank, (sid, score) in enumerate(scores, start=1):
                rows.append(
                    {
                        "record_id": rid,
                        "text_mode": TEXT_MODE,
                        "candidate_mode": candidate_mode,
                        "rank": rank,
                        "synset_id": sid,
                        "score": score,
                    }
                )

    return pd.DataFrame(rows)


def evaluate_topk(rankings: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Evaluate top-k EQUAL F1 for each candidate mode."""
    gold_eval = gold.copy()
    gold_eval["record_id"] = gold_eval["record_id"].astype(int)
    gold_eval["synset_id"] = gold_eval["synset_id"].astype(str)
    gold_eval["label_bin"] = (gold_eval["label"].astype(str) == "EQUAL").astype(int)

    rows = []

    for candidate_mode in sorted(rankings["candidate_mode"].unique()):
        ranking_mode = rankings[rankings["candidate_mode"] == candidate_mode].copy()

        for topk in TOPKS:
            pred_pairs = ranking_mode[ranking_mode["rank"] <= topk][["record_id", "synset_id"]].copy()
            pred_pairs["pred_bin"] = 1

            merged = gold_eval.merge(pred_pairs, on=["record_id", "synset_id"], how="left")
            merged["pred_bin"] = merged["pred_bin"].fillna(0).astype(int)

            tp = int(((merged["label_bin"] == 1) & (merged["pred_bin"] == 1)).sum())
            fp = int(((merged["label_bin"] == 0) & (merged["pred_bin"] == 1)).sum())
            fn = int(((merged["label_bin"] == 1) & (merged["pred_bin"] == 0)).sum())

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            rows.append(
                {
                    "candidate_mode": candidate_mode,
                    "topk": topk,
                    "precision_equal": precision,
                    "recall_equal": recall,
                    "f1_equal": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

    return pd.DataFrame(rows).sort_values(["f1_equal", "precision_equal"], ascending=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = pd.read_pickle(RECORDS_PATH)
    gold = pd.read_pickle(GOLD_PATH)
    babelnet = pd.read_pickle(BABELNET_PATH)
    babelnet.index = babelnet.index.map(str)

    rankings = build_rankings(records, gold, babelnet)
    rankings.to_pickle(RANKINGS_PATH)
    print(f"Saved rankings: {RANKINGS_PATH}")

    metrics = evaluate_topk(rankings, gold)
    with METRICS_PATH.open("wb") as f:
        pickle.dump({"summary": metrics}, f)
    print(f"Saved metrics: {METRICS_PATH}")

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
