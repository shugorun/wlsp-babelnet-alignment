# ベースライン実験ログ（日本語版）

> **注意**: このファイルは [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)（英語版）と同じ実験を日本語で記録したものです。  
> 実験の正典（canonical）は英語版です。内容に差異がある場合は英語版を優先してください。

このファイルは、[src/baselines/](../../src/baselines/) で再実行可能なベースライン実験の流れを日本語で記録したものです。

- 対象: [src/baselines/](../../src/baselines/)
- 出力先: [outputs/baselines/](../../outputs/baselines/)
- 補足: `cross-encoder` reranker のスコアは追加学習なしの top-k 比較として扱う。ただし、Gold B の hybrid 設定選択には使っていない。

---

## 2026-04-07 16:21:00 | gold_B_pairwise_baseline

- スクリプト: [src/baselines/run_gold_b_baseline.py](../../src/baselines/run_gold_b_baseline.py)
- 目的: 後続の ranking / hybrid の比較基準として、`gold_B` 上の lexical / pairwise ベースラインを再構築する
- パラメータ:
  - `candidate_columns`: `sids_JA,sids_JMdict`
  - `positive_label`: `EQUAL`
  - `model_weighted`: `rule_weighted_scoring`
  - `model_logreg`: `LogisticRegression`
  - `n_splits`: `5`
  - `random_state`: `0`
- 結果:
  - `weighted_pair_f1`: `0.540984`
  - `logreg_pair_f1`: `0.545918`
  - `logreg_record_exact_match_rate`: `0.412429`
- 出力:
  - [outputs/baselines/gold_B/pair_features.pkl](../../outputs/baselines/gold_B/pair_features.pkl)
  - [outputs/baselines/gold_B/oof_predictions.pkl](../../outputs/baselines/gold_B/oof_predictions.pkl)
  - [outputs/baselines/gold_B/metrics.pkl](../../outputs/baselines/gold_B/metrics.pkl)
  - [outputs/baselines/gold_B/logreg_model.pkl](../../outputs/baselines/gold_B/logreg_model.pkl)

---

## 2026-04-07 16:22:11 | gold_B_ranking_e5

- スクリプト: [src/baselines/run_gold_b_ranking.py](../../src/baselines/run_gold_b_ranking.py)
- 目的: `gold_B` 上で E5 ranking ベースラインを再構築し、複数の query / passage テンプレートを比較する
- パラメータ:
  - `model_name`: `intfloat/multilingual-e5-large`
  - `n_query_modes`: `8`
  - `n_passage_modes`: `7`
  - `n_text_modes`: `32`
  - `candidate_modes`: `union,ja_preferred`
  - `topks`: `1,2,3,5`
  - `max_length`: `384`
- 結果:
  - `best_text_mode`: `natural`
  - `best_query_mode`: `natural`
  - `best_passage_mode`: `natural`
  - `best_candidate_mode`: `ja_preferred`
  - `best_topk`: `1`
  - `best_pair_f1`: `0.657459`
  - `best_pair_precision`: `0.672316`
  - `best_pair_recall`: `0.643243`
- 出力:
  - [outputs/baselines/gold_B_ranking/rankings.pkl](../../outputs/baselines/gold_B_ranking/rankings.pkl)
  - [outputs/baselines/gold_B_ranking/metrics.pkl](../../outputs/baselines/gold_B_ranking/metrics.pkl)

---

## 2026-04-07 16:23:21 | gold_B_hybrid_search

- スクリプト: [src/baselines/run_gold_b_hybrid_search.py](../../src/baselines/run_gold_b_hybrid_search.py)
- 目的: ranking テンプレートと複数の feature preset を組み合わせて、再現可能な hybrid 設定を探索する
- パラメータ:
  - `n_splits`: `5`
  - `random_state`: `0`
  - `n_text_modes`: `32`
  - `feature_presets`: `full_both_top2,ja_preferred_top1,ja_preferred_top2,compact_both_top1`
- 結果:
  - `best_text_mode`: `natural`
  - `best_query_mode`: `natural`
  - `best_passage_mode`: `natural`
  - `best_feature_preset`: `ja_preferred_top2`
  - `best_pair_f1`: `0.684466`
  - `best_pair_precision`: `0.621145`
  - `best_pair_recall`: `0.762162`
  - `best_record_exact_match_rate`: `0.474576`
- 出力:
  - [outputs/baselines/gold_B_hybrid_search/summary.pkl](../../outputs/baselines/gold_B_hybrid_search/summary.pkl)
  - [outputs/baselines/gold_B_hybrid_best/pair_features_hybrid.pkl](../../outputs/baselines/gold_B_hybrid_best/pair_features_hybrid.pkl)
  - [outputs/baselines/gold_B_hybrid_best/oof_predictions.pkl](../../outputs/baselines/gold_B_hybrid_best/oof_predictions.pkl)
  - [outputs/baselines/gold_B_hybrid_best/metrics.pkl](../../outputs/baselines/gold_B_hybrid_best/metrics.pkl)
  - [outputs/baselines/gold_B_hybrid_best/logreg_model.pkl](../../outputs/baselines/gold_B_hybrid_best/logreg_model.pkl)
  - [outputs/baselines/gold_B_hybrid_best/config.json](../../outputs/baselines/gold_B_hybrid_best/config.json)

---

## 2026-04-07 16:38:26 | gold_B_record_decoder

- スクリプト: [src/baselines/run_gold_b_record_decoder.py](../../src/baselines/run_gold_b_record_decoder.py)
- 目的: best hybrid の pairwise スコアを record 単位に復号し、exact match 寄りの出力制御を比較する
- パラメータ:
  - `input_hybrid_dir`: [outputs/baselines/gold_B_hybrid_best](../../outputs/baselines/gold_B_hybrid_best)
  - `n_splits`: `5`
  - `random_state`: `0`
  - `param_grid_max_k`: `1,2,3`
  - `param_grid_score_min`: `0.5,0.6,0.7,0.8,0.85`
  - `param_grid_ratio_min`: `0.6,0.7,0.8,0.9,0.95`
- 結果:
  - `pair_f1`: `0.661290`
  - `record_macro_f1`: `0.616196`
  - `record_exact_match_rate`: `0.491525`
- 出力:
  - [outputs/baselines/gold_B_record_decoder/oof_predictions.pkl](../../outputs/baselines/gold_B_record_decoder/oof_predictions.pkl)
  - [outputs/baselines/gold_B_record_decoder/metrics.pkl](../../outputs/baselines/gold_B_record_decoder/metrics.pkl)
  - [outputs/baselines/gold_B_record_decoder/logreg_model.pkl](../../outputs/baselines/gold_B_record_decoder/logreg_model.pkl)

---

## 2026-04-07 17:07:29 | gold_A_hybrid_best_apply

- スクリプト: [src/baselines/run_gold_a_hybrid_best.py](../../src/baselines/run_gold_a_hybrid_best.py)
- 目的: `gold_B` で得た再現可能な best hybrid 設定を、そのまま `gold_A` に転移・評価する
- パラメータ:
  - `source_gold_B_config`: [outputs/baselines/gold_B_hybrid_best/config.json](../../outputs/baselines/gold_B_hybrid_best/config.json)
  - `query_mode`: `natural`
  - `passage_mode`: `natural`
  - `candidate_mode`: `ja_preferred`
  - `n_features`: `21`
  - `threshold`: `0.794092`
- 結果:
  - `pair_f1_equal`: `0.531915`
  - `pair_precision_equal`: `0.423729`
  - `pair_recall_equal`: `0.714286`
  - `record_macro_f1`: `0.456977`
  - `record_exact_match_rate`: `0.302326`
- 出力:
  - [outputs/baselines/gold_A_hybrid_best/config.json](../../outputs/baselines/gold_A_hybrid_best/config.json)
  - [outputs/baselines/gold_A_hybrid_best/pair_features.pkl](../../outputs/baselines/gold_A_hybrid_best/pair_features.pkl)
  - [outputs/baselines/gold_A_hybrid_best/rankings.pkl](../../outputs/baselines/gold_A_hybrid_best/rankings.pkl)
  - [outputs/baselines/gold_A_hybrid_best/predictions.pkl](../../outputs/baselines/gold_A_hybrid_best/predictions.pkl)
  - [outputs/baselines/gold_A_hybrid_best/pair_predictions.csv](../../outputs/baselines/gold_A_hybrid_best/pair_predictions.csv)
  - [outputs/baselines/gold_A_hybrid_best/summary.json](../../outputs/baselines/gold_A_hybrid_best/summary.json)

---

## 2026-04-07 17:13:12 | gold_A_no_training_methods

- スクリプト: [src/baselines/run_gold_a_no_training_methods.py](../../src/baselines/run_gold_a_no_training_methods.py)
- 目的: `gold_B` を学習に使わない手法だけを `gold_A` 上で比較する
- 比較対象:
  - weighted lexical ranking
  - E5 ranking
  - MPNet ranking
  - pretrained cross-encoder
- パラメータ:
  - `topks`: `1,2,3,5`
  - `methods`: `weighted_lexical,e5_ranking,mpnet_ranking,cross_encoder_pretrained`
  - `cross_model_name`: `BAAI/bge-reranker-v2-m3`
  - `mpnet_text_mode`: `concise`
  - `cross_text_mode`: `concise`
- 結果:
  - `best_method`: `cross_encoder_pretrained`
  - `best_text_mode`: `concise`
  - `best_candidate_mode`: `ja_preferred`
  - `best_topk`: `1`
  - `best_pair_f1_equal`: `0.551282`
  - `best_pair_precision_equal`: `0.500000`
  - `best_pair_recall_equal`: `0.614286`
- 出力:
  - [outputs/baselines/gold_A_no_training_methods/weighted_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/weighted_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/e5_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/e5_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/mpnet_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/mpnet_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/cross_encoder_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/cross_encoder_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/summary.pkl](../../outputs/baselines/gold_A_no_training_methods/summary.pkl)
  - [outputs/baselines/gold_A_no_training_methods/summary.csv](../../outputs/baselines/gold_A_no_training_methods/summary.csv)

---

## 2026-04-12 19:29:05 | gold_B_cross_encoder_no_training_topk

- スクリプト: [src/baselines/run_gold_b_cross_encoder.py](../../src/baselines/run_gold_b_cross_encoder.py)
- 目的: 事前学習済み BGE reranker を `gold_B` 上で追加学習なし・top-k 復号のみで評価し、`gold_A` の no-training reranker 比較に対応する `gold_B` 側の結果を残す
- パラメータ:
  - `model_name`: `BAAI/bge-reranker-v2-m3`
  - `text_mode`: `concise`
  - `topks`: `1,2,3,5`
  - `decoding`: `top-k by reranker score`
- 結果:
  - `best_topk`: `1`
  - `best_pair_f1`: `0.585635`
  - `best_pair_precision`: `0.598870`
  - `best_pair_recall`: `0.572973`
  - `best_record_exact_match_rate`: `0.435028`
- 出力:
  - [outputs/baselines/gold_B_cross_encoder/oof_predictions.pkl](../../outputs/baselines/gold_B_cross_encoder/oof_predictions.pkl)
  - [outputs/baselines/gold_B_cross_encoder/topk_summary.pkl](../../outputs/baselines/gold_B_cross_encoder/topk_summary.pkl)
  - [outputs/baselines/gold_B_cross_encoder/topk_summary.csv](../../outputs/baselines/gold_B_cross_encoder/topk_summary.csv)
