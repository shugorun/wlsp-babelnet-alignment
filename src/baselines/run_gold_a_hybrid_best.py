from __future__ import annotations

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.experiment_log import append_run_log
from baselines.run_gold_b_baseline import build_feature_rows
from baselines.run_gold_b_ranking import DTYPE, DEVICE, build_passage_text, build_query_text, encode_texts

ROOT = Path(__file__).resolve().parents[2]

GOLD_BEST_DIR = ROOT / 'outputs' / 'baselines' / 'gold_B_hybrid_best'
CONFIG_PATH = GOLD_BEST_DIR / 'config.json'
MODEL_PATH = GOLD_BEST_DIR / 'logreg_model.pkl'

RECORDS_PATH = ROOT / 'data' / 'gold' / 'gold_A_records.pkl'
GOLD_PATH = ROOT / 'data' / 'gold' / 'gold_A.pkl'
BABELNET_PATH = ROOT / 'data' / 'processed' / 'babelnet_.pkl'

OUTPUT_DIR = ROOT / 'outputs' / 'baselines' / 'gold_A_hybrid_best'
PAIR_FEATURES_PATH = OUTPUT_DIR / 'pair_features.pkl'
RANKINGS_PATH = OUTPUT_DIR / 'rankings.pkl'
PREDICTIONS_PATH = OUTPUT_DIR / 'predictions.pkl'
PREDICTIONS_CSV_PATH = OUTPUT_DIR / 'pair_predictions.csv'
SUMMARY_PATH = OUTPUT_DIR / 'summary.json'
OUT_CONFIG_PATH = OUTPUT_DIR / 'config.json'


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
    return list(dict.fromkeys(out))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def record_level_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    exact_matches = []

    for _, group in df.groupby('record_id', sort=False):
        gold_set = set(group.loc[group['label'] == 1, 'synset_id'])
        pred_set = set(group.loc[group[pred_col] == 1, 'synset_id'])

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
        'record_macro_precision': float(np.mean(precisions)),
        'record_macro_recall': float(np.mean(recalls)),
        'record_macro_f1': float(np.mean(f1s)),
        'record_exact_match_rate': float(np.mean(exact_matches)),
    }


def build_gold_a_rankings(records: pd.DataFrame, gold: pd.DataFrame, babelnet: pd.DataFrame, query_mode: str, passage_mode: str, candidate_mode: str) -> pd.DataFrame:
    """Rebuild only the ranking features needed by the selected gold_B config."""
    model_name = 'intfloat/multilingual-e5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, dtype=DTYPE).to(DEVICE).eval()

    record_ids = records.index.astype(int).tolist()
    candidate_sids = sorted({str(sid) for sid in gold['synset_id'].astype(str).tolist()})

    query_texts = [build_query_text(records.loc[rid], query_mode) for rid in record_ids]
    query_vecs = encode_texts(model, tokenizer, query_texts)
    query_vec_by_rid = {rid: query_vecs[i] for i, rid in enumerate(record_ids)}

    passage_texts = [build_passage_text(babelnet.loc[sid], passage_mode) for sid in candidate_sids]
    passage_vecs = encode_texts(model, tokenizer, passage_texts)
    passage_vec_by_sid = {sid: passage_vecs[i] for i, sid in enumerate(candidate_sids)}

    rows = []
    for rid, row in records.iterrows():
        rid = int(rid)
        query_vec = query_vec_by_rid[rid]

        sids_ja = as_str_list(row.get('sids_JA', []))
        sids_jm = as_str_list(row.get('sids_JMdict', []))
        if candidate_mode == 'ja_preferred':
            candidate_sids_for_row = sids_ja if sids_ja else sids_jm
        elif candidate_mode == 'union':
            candidate_sids_for_row = list(dict.fromkeys(sids_ja + sids_jm))
        else:
            raise ValueError(f'Unknown candidate_mode: {candidate_mode}')

        filtered = [sid for sid in candidate_sids_for_row if sid in passage_vec_by_sid]
        if not filtered:
            continue

        scores = []
        for sid in filtered:
            score = float(np.dot(query_vec, passage_vec_by_sid[sid]))
            scores.append((sid, score))
        scores.sort(key=lambda item: item[1], reverse=True)

        for rank, (sid, score) in enumerate(scores, start=1):
            rows.append({
                'record_id': rid,
                'text_mode': f'{query_mode}__{passage_mode}' if query_mode != passage_mode else query_mode,
                'query_mode': query_mode,
                'passage_mode': passage_mode,
                'candidate_mode': candidate_mode,
                'rank': rank,
                'synset_id': sid,
                'score': score,
            })

    return pd.DataFrame(rows)


def build_ranking_feature_table(rankings: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for mode in ['union', 'ja_preferred']:
        df_mode = rankings[rankings['candidate_mode'] == mode].copy()
        if df_mode.empty:
            continue
        df_mode = df_mode.rename(columns={'score': f'score_{mode}', 'rank': f'rank_{mode}'})
        df_mode[f'inv_rank_{mode}'] = 1.0 / df_mode[f'rank_{mode}']
        df_mode[f'in_top1_{mode}'] = (df_mode[f'rank_{mode}'] <= 1).astype(int)
        df_mode[f'in_top2_{mode}'] = (df_mode[f'rank_{mode}'] <= 2).astype(int)
        keep = ['record_id', 'synset_id', f'score_{mode}', f'rank_{mode}', f'inv_rank_{mode}', f'in_top1_{mode}', f'in_top2_{mode}']
        parts.append(df_mode[keep])

    out = None
    for part in parts:
        out = part if out is None else out.merge(part, on=['record_id', 'synset_id'], how='outer')

    if out is None:
        out = pd.DataFrame(columns=['record_id', 'synset_id'])
    out['record_id'] = out['record_id'].astype(int)
    out['synset_id'] = out['synset_id'].astype(str)
    return out


def merge_features(pair_df: pd.DataFrame, ranking_features: pd.DataFrame) -> pd.DataFrame:
    df = pair_df.merge(ranking_features, on=['record_id', 'synset_id'], how='left')
    return df.fillna({
        'score_union': -1.0,
        'rank_union': 999.0,
        'inv_rank_union': 0.0,
        'in_top1_union': 0,
        'in_top2_union': 0,
        'score_ja_preferred': -1.0,
        'rank_ja_preferred': 999.0,
        'inv_rank_ja_preferred': 0.0,
        'in_top1_ja_preferred': 0,
        'in_top2_ja_preferred': 0,
    })


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the reproducible best config learned on gold_B and apply the same
    # feature space and threshold to gold_A.
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    with MODEL_PATH.open('rb') as f:
        model_bundle = pickle.load(f)

    records = pd.read_pickle(RECORDS_PATH)
    gold = pd.read_pickle(GOLD_PATH)
    babelnet = pd.read_pickle(BABELNET_PATH)
    babelnet.index = babelnet.index.map(str)

    query_mode = config.get('query_mode', config['text_mode'])
    passage_mode = config.get('passage_mode', config['text_mode'])
    candidate_mode = 'ja_preferred'
    feature_columns = config['feature_columns']
    threshold = float(model_bundle['threshold'])

    pair_df = build_feature_rows(records, gold, babelnet)
    pair_df.to_pickle(PAIR_FEATURES_PATH)

    # Only the ranking mode referenced by the saved gold_B config is rebuilt.
    rankings = build_gold_a_rankings(records, gold, babelnet, query_mode, passage_mode, candidate_mode)
    rankings.to_pickle(RANKINGS_PATH)

    ranking_features = build_ranking_feature_table(rankings)
    test_df = merge_features(pair_df, ranking_features)

    X_test = test_df[feature_columns].to_numpy(dtype=float)
    scores = model_bundle['model'].predict_proba(X_test)[:, 1]

    pred_df = test_df[['record_id', 'synset_id', 'label', 'raw_label']].copy()
    pred_df['score'] = scores
    pred_df['pred'] = (scores >= threshold).astype(int)
    pred_df.to_pickle(PREDICTIONS_PATH)
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding='utf-8-sig')

    pair_metrics = binary_metrics(pred_df['label'].to_numpy(dtype=int), pred_df['pred'].to_numpy(dtype=int))
    record_metrics = record_level_metrics(pred_df, 'pred')

    summary = {
        'config_name': 'gold_A_hybrid_best',
        'source_gold_B_config': str(CONFIG_PATH.relative_to(ROOT)),
        'query_mode': query_mode,
        'passage_mode': passage_mode,
        'candidate_mode': candidate_mode,
        'feature_columns': feature_columns,
        'threshold': threshold,
        'n_pairs': int(len(pred_df)),
        'n_records': int(pred_df['record_id'].nunique()),
        'pair_precision_equal': pair_metrics['precision'],
        'pair_recall_equal': pair_metrics['recall'],
        'pair_f1_equal': pair_metrics['f1'],
        'pair_accuracy_equal_vs_rest': pair_metrics['accuracy'],
        'record_macro_precision': record_metrics['record_macro_precision'],
        'record_macro_recall': record_metrics['record_macro_recall'],
        'record_macro_f1': record_metrics['record_macro_f1'],
        'record_exact_match_rate': record_metrics['record_exact_match_rate'],
        'tp': pair_metrics['tp'],
        'fp': pair_metrics['fp'],
        'fn': pair_metrics['fn'],
        'tn': pair_metrics['tn'],
    }

    OUT_CONFIG_PATH.write_text(json.dumps({
        'config_name': 'gold_A_hybrid_best',
        'source_gold_B_config': str(CONFIG_PATH.relative_to(ROOT)),
        'query_mode': query_mode,
        'passage_mode': passage_mode,
        'candidate_mode': candidate_mode,
        'feature_columns': feature_columns,
        'threshold': threshold,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    append_run_log(
        run_name='gold_A_hybrid_best_apply',
        rationale='Apply the best reproducible gold_B hybrid configuration to gold_A with the same feature columns, threshold, and ranking setup.',
        script_path='src/baselines/run_gold_a_hybrid_best.py',
        params={
            'source_gold_B_config': str(CONFIG_PATH.relative_to(ROOT)),
            'query_mode': query_mode,
            'passage_mode': passage_mode,
            'candidate_mode': candidate_mode,
            'n_features': len(feature_columns),
            'threshold': threshold,
        },
        metrics={
            'pair_f1_equal': summary['pair_f1_equal'],
            'pair_precision_equal': summary['pair_precision_equal'],
            'pair_recall_equal': summary['pair_recall_equal'],
            'record_macro_f1': summary['record_macro_f1'],
            'record_exact_match_rate': summary['record_exact_match_rate'],
        },
        outputs=[
            str(OUT_CONFIG_PATH.relative_to(ROOT)),
            str(PAIR_FEATURES_PATH.relative_to(ROOT)),
            str(RANKINGS_PATH.relative_to(ROOT)),
            str(PREDICTIONS_PATH.relative_to(ROOT)),
            str(PREDICTIONS_CSV_PATH.relative_to(ROOT)),
            str(SUMMARY_PATH.relative_to(ROOT)),
        ],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
