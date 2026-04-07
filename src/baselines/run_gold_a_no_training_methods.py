from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.experiment_log import append_run_log
from baselines.run_gold_b_baseline import build_feature_rows, weighted_score
from baselines.run_gold_b_ranking import build_rankings as build_e5_rankings
from baselines.run_gold_b_ranking import build_passage_text, build_query_text
from baselines.run_gold_b_ranking_mpnet import MODEL_DIR as MPNET_MODEL_DIR

ROOT = Path(__file__).resolve().parents[2]

RECORDS_PATH = ROOT / 'data' / 'gold' / 'gold_A_records.pkl'
GOLD_PATH = ROOT / 'data' / 'gold' / 'gold_A.pkl'
BABELNET_PATH = ROOT / 'data' / 'processed' / 'babelnet_.pkl'

OUTPUT_DIR = ROOT / 'outputs' / 'baselines' / 'gold_A_no_training_methods'
WEIGHTED_RANKINGS_PATH = OUTPUT_DIR / 'weighted_rankings.pkl'
E5_RANKINGS_PATH = OUTPUT_DIR / 'e5_rankings.pkl'
MPNET_RANKINGS_PATH = OUTPUT_DIR / 'mpnet_rankings.pkl'
CROSS_RANKINGS_PATH = OUTPUT_DIR / 'cross_encoder_rankings.pkl'
SUMMARY_PATH = OUTPUT_DIR / 'summary.pkl'
SUMMARY_CSV_PATH = OUTPUT_DIR / 'summary.csv'

TOPKS = [1, 2, 3, 5]
MPNET_TEXT_MODE = 'concise'
CROSS_TEXT_MODE = 'concise'
CROSS_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
CROSS_MAX_LENGTH = 512
CROSS_BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# This script keeps all methods training-free with respect to gold_B.
# The only tuning knob is top-k decoding over the fixed candidate set.

def dedupe_keep_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, (list, tuple, set, pd.Series, pd.Index)):
        value = [value]

    out = []
    for item in value:
        if item is None or pd.isna(item):
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return dedupe_keep_order(out)


def add_candidate_membership(pair_df: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    has_ja_map = {}
    for rid, row in records.iterrows():
        has_ja_map[int(rid)] = len(as_str_list(row.get('sids_JA', []))) > 0

    out['eligible_union'] = ((out['from_sids_JA'] == 1) | (out['from_sids_JMdict'] == 1)).astype(int)
    out['record_has_sids_JA'] = out['record_id'].map(has_ja_map).fillna(False)
    out['eligible_ja_preferred'] = np.where(out['record_has_sids_JA'], out['from_sids_JA'], out['from_sids_JMdict']).astype(int)
    return out


def evaluate_rankings(rankings: pd.DataFrame, gold: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    gold_eval = gold.copy()
    gold_eval['record_id'] = gold_eval['record_id'].astype(int)
    gold_eval['synset_id'] = gold_eval['synset_id'].astype(str)
    gold_eval['label_bin'] = (gold_eval['label'].astype(str) == 'EQUAL').astype(int)

    rows = []
    unique_groups = rankings[group_columns].drop_duplicates().to_dict(orient='records')
    for group in unique_groups:
        mask = np.ones(len(rankings), dtype=bool)
        for col, val in group.items():
            mask &= rankings[col].eq(val).to_numpy()
        ranking_mode = rankings.loc[mask].copy()

        for topk in TOPKS:
            pred_pairs = ranking_mode[ranking_mode['rank'] <= topk][['record_id', 'synset_id']].copy()
            pred_pairs['pred_bin'] = 1

            merged = gold_eval.merge(pred_pairs, on=['record_id', 'synset_id'], how='left')
            merged['pred_bin'] = merged['pred_bin'].fillna(0).astype(int)

            tp = int(((merged['label_bin'] == 1) & (merged['pred_bin'] == 1)).sum())
            fp = int(((merged['label_bin'] == 0) & (merged['pred_bin'] == 1)).sum())
            fn = int(((merged['label_bin'] == 1) & (merged['pred_bin'] == 0)).sum())

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            row = dict(group)
            row.update({
                'topk': topk,
                'precision_equal': precision,
                'recall_equal': recall,
                'f1_equal': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
            })
            rows.append(row)

    return pd.DataFrame(rows).sort_values(['f1_equal', 'precision_equal', 'recall_equal'], ascending=False)


def build_weighted_rankings(records: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """Rank gold_A candidates with the hand-crafted lexical score only."""
    df = add_candidate_membership(pair_df, records)
    df['score'] = weighted_score(df)

    rows = []
    for candidate_mode, eligible_col in [('union', 'eligible_union'), ('ja_preferred', 'eligible_ja_preferred')]:
        subset = df[df[eligible_col] == 1].copy()
        for rid, group in subset.groupby('record_id', sort=False):
            ranked = group.sort_values(['score', 'synset_id'], ascending=[False, True])
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                rows.append({
                    'method': 'weighted_lexical',
                    'text_mode': 'weighted_score',
                    'candidate_mode': candidate_mode,
                    'record_id': int(row.record_id),
                    'synset_id': str(row.synset_id),
                    'rank': rank,
                    'score': float(row.score),
                })
    return pd.DataFrame(rows)


def build_mpnet_rankings(records: pd.DataFrame, gold: pd.DataFrame, babelnet: pd.DataFrame) -> pd.DataFrame:
    """Rank with a sentence-transformer embedding model without task training."""
    model = SentenceTransformer(str(MPNET_MODEL_DIR))

    record_ids = records.index.astype(int).tolist()
    query_texts = [build_query_text(records.loc[rid], MPNET_TEXT_MODE) for rid in record_ids]
    query_vecs = model.encode(query_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    query_vec_by_rid = {rid: query_vecs[i] for i, rid in enumerate(record_ids)}

    candidate_sids = sorted({str(sid) for sid in gold['synset_id'].astype(str).tolist()})
    passage_texts = [build_passage_text(babelnet.loc[sid], MPNET_TEXT_MODE) for sid in candidate_sids]
    passage_vecs = model.encode(passage_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    passage_vec_by_sid = {sid: passage_vecs[i] for i, sid in enumerate(candidate_sids)}

    rows = []
    for rid, row in records.iterrows():
        rid = int(rid)
        query_vec = query_vec_by_rid[rid]
        sids_ja = as_str_list(row.get('sids_JA', []))
        sids_jm = as_str_list(row.get('sids_JMdict', []))

        candidate_specs = {
            'union': dedupe_keep_order(sids_ja + sids_jm),
            'ja_preferred': sids_ja if sids_ja else sids_jm,
        }
        for candidate_mode, candidate_sids_for_row in candidate_specs.items():
            filtered = [sid for sid in candidate_sids_for_row if sid in passage_vec_by_sid]
            if not filtered:
                continue
            scores = [(sid, float(np.dot(query_vec, passage_vec_by_sid[sid]))) for sid in filtered]
            scores.sort(key=lambda item: item[1], reverse=True)
            for rank, (sid, score) in enumerate(scores, start=1):
                rows.append({
                    'method': 'mpnet_ranking',
                    'text_mode': MPNET_TEXT_MODE,
                    'candidate_mode': candidate_mode,
                    'record_id': rid,
                    'synset_id': sid,
                    'rank': rank,
                    'score': score,
                })
    return pd.DataFrame(rows)


def build_cross_pair_table(records: pd.DataFrame, gold: pd.DataFrame, babelnet: pd.DataFrame) -> pd.DataFrame:
    """Materialize query/passage pairs for the pretrained reranker."""
    rows = []
    for pair in gold.itertuples(index=False):
        rid = int(pair.record_id)
        sid = str(pair.synset_id)
        record = records.loc[rid]
        synset = babelnet.loc[sid]
        rows.append({
            'record_id': rid,
            'synset_id': sid,
            'query_text': build_query_text(record, CROSS_TEXT_MODE),
            'passage_text': build_passage_text(synset, CROSS_TEXT_MODE),
        })
    return pd.DataFrame(rows)


def score_cross_pairs(pair_df: pd.DataFrame) -> np.ndarray:
    """Score pairs directly with a pretrained cross-encoder."""
    tokenizer = AutoTokenizer.from_pretrained(CROSS_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(CROSS_MODEL_NAME).to(DEVICE)
    model.eval()

    scores = []
    total = len(pair_df)
    for start in tqdm(range(0, total, CROSS_BATCH_SIZE), desc='Cross scoring', unit='batch'):
        batch = pair_df.iloc[start:start + CROSS_BATCH_SIZE]
        encoded = tokenizer(
            batch['query_text'].tolist(),
            batch['passage_text'].tolist(),
            padding=True,
            truncation=True,
            max_length=CROSS_MAX_LENGTH,
            return_tensors='pt',
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        scores.append(logits.view(-1).detach().cpu().numpy())

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=float)


def build_cross_rankings(records: pd.DataFrame, pair_df: pd.DataFrame, gold: pd.DataFrame, babelnet: pd.DataFrame) -> pd.DataFrame:
    cross_pairs = build_cross_pair_table(records, gold, babelnet)
    cross_pairs['score'] = score_cross_pairs(cross_pairs)
    cross_pairs = cross_pairs.merge(pair_df[['record_id', 'synset_id', 'from_sids_JA', 'from_sids_JMdict']], on=['record_id', 'synset_id'], how='left')
    cross_pairs = add_candidate_membership(cross_pairs, records)

    rows = []
    for candidate_mode, eligible_col in [('union', 'eligible_union'), ('ja_preferred', 'eligible_ja_preferred')]:
        subset = cross_pairs[cross_pairs[eligible_col] == 1].copy()
        for rid, group in subset.groupby('record_id', sort=False):
            ranked = group.sort_values(['score', 'synset_id'], ascending=[False, True])
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                rows.append({
                    'method': 'cross_encoder_pretrained',
                    'text_mode': CROSS_TEXT_MODE,
                    'candidate_mode': candidate_mode,
                    'record_id': int(row.record_id),
                    'synset_id': str(row.synset_id),
                    'rank': rank,
                    'score': float(row.score),
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build one shared pair table, then compare several no-training ranking
    # strategies under the same top-k evaluation protocol.
    records = pd.read_pickle(RECORDS_PATH)
    gold = pd.read_pickle(GOLD_PATH)
    babelnet = pd.read_pickle(BABELNET_PATH)
    babelnet.index = babelnet.index.map(str)

    pair_df = build_feature_rows(records, gold, babelnet)

    weighted_rankings = build_weighted_rankings(records, pair_df)
    weighted_rankings.to_pickle(WEIGHTED_RANKINGS_PATH)
    weighted_summary = evaluate_rankings(weighted_rankings, gold, ['method', 'text_mode', 'candidate_mode'])

    e5_rankings = build_e5_rankings(records, gold, babelnet)
    e5_rankings['method'] = 'e5_ranking'
    e5_rankings.to_pickle(E5_RANKINGS_PATH)
    e5_summary = evaluate_rankings(e5_rankings, gold, ['method', 'text_mode', 'query_mode', 'passage_mode', 'candidate_mode'])

    mpnet_rankings = build_mpnet_rankings(records, gold, babelnet)
    mpnet_rankings.to_pickle(MPNET_RANKINGS_PATH)
    mpnet_summary = evaluate_rankings(mpnet_rankings, gold, ['method', 'text_mode', 'candidate_mode'])

    cross_rankings = build_cross_rankings(records, pair_df, gold, babelnet)
    cross_rankings.to_pickle(CROSS_RANKINGS_PATH)
    cross_summary = evaluate_rankings(cross_rankings, gold, ['method', 'text_mode', 'candidate_mode'])

    summary = pd.concat([weighted_summary, e5_summary, mpnet_summary, cross_summary], ignore_index=True, sort=False)
    summary = summary.sort_values(['f1_equal', 'precision_equal', 'recall_equal'], ascending=False)
    summary.to_pickle(SUMMARY_PATH)
    summary.to_csv(SUMMARY_CSV_PATH, index=False, encoding='utf-8-sig')

    best_row = summary.iloc[0].to_dict()
    append_run_log(
        run_name='gold_A_no_training_methods',
        rationale='Compare methods that do not learn from gold_B: weighted lexical ranking, E5 ranking, MPNet ranking, and a pretrained cross-encoder, all evaluated on gold_A with top-k decoding only.',
        script_path='src/baselines/run_gold_a_no_training_methods.py',
        params={
            'topks': ','.join(str(k) for k in TOPKS),
            'methods': 'weighted_lexical,e5_ranking,mpnet_ranking,cross_encoder_pretrained',
            'cross_model_name': CROSS_MODEL_NAME,
            'mpnet_text_mode': MPNET_TEXT_MODE,
            'cross_text_mode': CROSS_TEXT_MODE,
        },
        metrics={
            'best_method': best_row.get('method', ''),
            'best_text_mode': best_row.get('text_mode', ''),
            'best_candidate_mode': best_row.get('candidate_mode', ''),
            'best_topk': best_row.get('topk', ''),
            'best_pair_f1_equal': best_row.get('f1_equal', 0.0),
            'best_pair_precision_equal': best_row.get('precision_equal', 0.0),
            'best_pair_recall_equal': best_row.get('recall_equal', 0.0),
        },
        outputs=[
            str(WEIGHTED_RANKINGS_PATH.relative_to(ROOT)),
            str(E5_RANKINGS_PATH.relative_to(ROOT)),
            str(MPNET_RANKINGS_PATH.relative_to(ROOT)),
            str(CROSS_RANKINGS_PATH.relative_to(ROOT)),
            str(SUMMARY_PATH.relative_to(ROOT)),
            str(SUMMARY_CSV_PATH.relative_to(ROOT)),
        ],
    )

    print(summary.head(30).to_string(index=False))


if __name__ == '__main__':
    main()
