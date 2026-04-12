# Baseline Experiment Log

This log records the reproducible baseline sequence under [src/baselines/](../../src/baselines/).

- Scope: [src/baselines/](../../src/baselines/)
- Outputs: [outputs/baselines/](../../outputs/baselines/)
- Note: `cross-encoder` reranker scores are included as a no-training top-k comparison. They are not used to select the Gold B hybrid configuration.

---

## 2026-04-07 16:21:00 | gold_B_pairwise_baseline

- Script: [src/baselines/run_gold_b_baseline.py](../../src/baselines/run_gold_b_baseline.py)
- Rationale: Rebuild the lexical / pairwise baseline on `gold_B` as the reference point for later ranking and hybrid runs.
- Params:
  - `candidate_columns`: `sids_JA,sids_JMdict`
  - `positive_label`: `EQUAL`
  - `model_weighted`: `rule_weighted_scoring`
  - `model_logreg`: `LogisticRegression`
  - `n_splits`: `5`
  - `random_state`: `0`
- Metrics:
  - `weighted_pair_f1`: `0.540984`
  - `logreg_pair_f1`: `0.545918`
  - `logreg_record_exact_match_rate`: `0.412429`
- Outputs:
  - [outputs/baselines/gold_B/pair_features.pkl](../../outputs/baselines/gold_B/pair_features.pkl)
  - [outputs/baselines/gold_B/oof_predictions.pkl](../../outputs/baselines/gold_B/oof_predictions.pkl)
  - [outputs/baselines/gold_B/metrics.pkl](../../outputs/baselines/gold_B/metrics.pkl)
  - [outputs/baselines/gold_B/logreg_model.pkl](../../outputs/baselines/gold_B/logreg_model.pkl)

---

## 2026-04-07 16:22:11 | gold_B_ranking_e5

- Script: [src/baselines/run_gold_b_ranking.py](../../src/baselines/run_gold_b_ranking.py)
- Rationale: Rebuild the E5 ranking baseline on `gold_B` and compare multiple query / passage templates, including templates inspired by `../final`.
- Params:
  - `model_name`: `intfloat/multilingual-e5-large`
  - `n_query_modes`: `8`
  - `n_passage_modes`: `7`
  - `n_text_modes`: `32`
  - `candidate_modes`: `union,ja_preferred`
  - `topks`: `1,2,3,5`
  - `max_length`: `384`
- Metrics:
  - `best_text_mode`: `natural`
  - `best_query_mode`: `natural`
  - `best_passage_mode`: `natural`
  - `best_candidate_mode`: `ja_preferred`
  - `best_topk`: `1`
  - `best_pair_f1`: `0.657459`
  - `best_pair_precision`: `0.672316`
  - `best_pair_recall`: `0.643243`
- Outputs:
  - [outputs/baselines/gold_B_ranking/rankings.pkl](../../outputs/baselines/gold_B_ranking/rankings.pkl)
  - [outputs/baselines/gold_B_ranking/metrics.pkl](../../outputs/baselines/gold_B_ranking/metrics.pkl)

---

## 2026-04-07 16:23:21 | gold_B_hybrid_search

- Script: [src/baselines/run_gold_b_hybrid_search.py](../../src/baselines/run_gold_b_hybrid_search.py)
- Rationale: Search reproducible hybrid settings by combining the ranking templates with several feature presets, then export the best configuration.
- Params:
  - `n_splits`: `5`
  - `random_state`: `0`
  - `n_text_modes`: `32`
  - `feature_presets`: `full_both_top2,ja_preferred_top1,ja_preferred_top2,compact_both_top1`
- Metrics:
  - `best_text_mode`: `natural`
  - `best_query_mode`: `natural`
  - `best_passage_mode`: `natural`
  - `best_feature_preset`: `ja_preferred_top2`
  - `best_pair_f1`: `0.684466`
  - `best_pair_precision`: `0.621145`
  - `best_pair_recall`: `0.762162`
  - `best_record_exact_match_rate`: `0.474576`
- Outputs:
  - [outputs/baselines/gold_B_hybrid_search/summary.pkl](../../outputs/baselines/gold_B_hybrid_search/summary.pkl)
  - [outputs/baselines/gold_B_hybrid_best/pair_features_hybrid.pkl](../../outputs/baselines/gold_B_hybrid_best/pair_features_hybrid.pkl)
  - [outputs/baselines/gold_B_hybrid_best/oof_predictions.pkl](../../outputs/baselines/gold_B_hybrid_best/oof_predictions.pkl)
  - [outputs/baselines/gold_B_hybrid_best/metrics.pkl](../../outputs/baselines/gold_B_hybrid_best/metrics.pkl)
  - [outputs/baselines/gold_B_hybrid_best/logreg_model.pkl](../../outputs/baselines/gold_B_hybrid_best/logreg_model.pkl)
  - [outputs/baselines/gold_B_hybrid_best/config.json](../../outputs/baselines/gold_B_hybrid_best/config.json)

---

## 2026-04-07 16:38:26 | gold_B_record_decoder

- Script: [src/baselines/run_gold_b_record_decoder.py](../../src/baselines/run_gold_b_record_decoder.py)
- Rationale: Decode the best hybrid pairwise scores at the record level and optimize for exact-match oriented output control.
- Params:
  - `input_hybrid_dir`: [outputs/baselines/gold_B_hybrid_best](../../outputs/baselines/gold_B_hybrid_best)
  - `n_splits`: `5`
  - `random_state`: `0`
  - `param_grid_max_k`: `1,2,3`
  - `param_grid_score_min`: `0.5,0.6,0.7,0.8,0.85`
  - `param_grid_ratio_min`: `0.6,0.7,0.8,0.9,0.95`
- Metrics:
  - `pair_f1`: `0.661290`
  - `record_macro_f1`: `0.616196`
  - `record_exact_match_rate`: `0.491525`
- Outputs:
  - [outputs/baselines/gold_B_record_decoder/oof_predictions.pkl](../../outputs/baselines/gold_B_record_decoder/oof_predictions.pkl)
  - [outputs/baselines/gold_B_record_decoder/metrics.pkl](../../outputs/baselines/gold_B_record_decoder/metrics.pkl)
  - [outputs/baselines/gold_B_record_decoder/logreg_model.pkl](../../outputs/baselines/gold_B_record_decoder/logreg_model.pkl)

---

## 2026-04-07 17:07:29 | gold_A_hybrid_best_apply

- Script: [src/baselines/run_gold_a_hybrid_best.py](../../src/baselines/run_gold_a_hybrid_best.py)
- Rationale: Apply the best reproducible gold_B hybrid configuration to gold_A with the same feature columns, threshold, and ranking setup.
- Params:
  - `source_gold_B_config`: [outputs/baselines/gold_B_hybrid_best/config.json](../../outputs/baselines/gold_B_hybrid_best/config.json)
  - `query_mode`: `natural`
  - `passage_mode`: `natural`
  - `candidate_mode`: `ja_preferred`
  - `n_features`: `21`
  - `threshold`: `0.794092`
- Metrics:
  - `pair_f1_equal`: `0.531915`
  - `pair_precision_equal`: `0.423729`
  - `pair_recall_equal`: `0.714286`
  - `record_macro_f1`: `0.456977`
  - `record_exact_match_rate`: `0.302326`
- Outputs:
  - [outputs/baselines/gold_A_hybrid_best/config.json](../../outputs/baselines/gold_A_hybrid_best/config.json)
  - [outputs/baselines/gold_A_hybrid_best/pair_features.pkl](../../outputs/baselines/gold_A_hybrid_best/pair_features.pkl)
  - [outputs/baselines/gold_A_hybrid_best/rankings.pkl](../../outputs/baselines/gold_A_hybrid_best/rankings.pkl)
  - [outputs/baselines/gold_A_hybrid_best/predictions.pkl](../../outputs/baselines/gold_A_hybrid_best/predictions.pkl)
  - [outputs/baselines/gold_A_hybrid_best/pair_predictions.csv](../../outputs/baselines/gold_A_hybrid_best/pair_predictions.csv)
  - [outputs/baselines/gold_A_hybrid_best/summary.json](../../outputs/baselines/gold_A_hybrid_best/summary.json)

---

## 2026-04-07 17:13:12 | gold_A_no_training_methods

- Script: [src/baselines/run_gold_a_no_training_methods.py](../../src/baselines/run_gold_a_no_training_methods.py)
- Rationale: Compare methods that do not learn from gold_B: weighted lexical ranking, E5 ranking, MPNet ranking, and a pretrained cross-encoder, all evaluated on gold_A with top-k decoding only.
- Params:
  - `topks`: `1,2,3,5`
  - `methods`: `weighted_lexical,e5_ranking,mpnet_ranking,cross_encoder_pretrained`
  - `cross_model_name`: `BAAI/bge-reranker-v2-m3`
  - `mpnet_text_mode`: `concise`
  - `cross_text_mode`: `concise`
- Metrics:
  - `best_method`: `cross_encoder_pretrained`
  - `best_text_mode`: `concise`
  - `best_candidate_mode`: `ja_preferred`
  - `best_topk`: `1`
  - `best_pair_f1_equal`: `0.551282`
  - `best_pair_precision_equal`: `0.500000`
  - `best_pair_recall_equal`: `0.614286`
- Outputs:
  - [outputs/baselines/gold_A_no_training_methods/weighted_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/weighted_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/e5_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/e5_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/mpnet_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/mpnet_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/cross_encoder_rankings.pkl](../../outputs/baselines/gold_A_no_training_methods/cross_encoder_rankings.pkl)
  - [outputs/baselines/gold_A_no_training_methods/summary.pkl](../../outputs/baselines/gold_A_no_training_methods/summary.pkl)
  - [outputs/baselines/gold_A_no_training_methods/summary.csv](../../outputs/baselines/gold_A_no_training_methods/summary.csv)

---

## 2026-04-12 19:29:05 | gold_B_cross_encoder_no_training_topk

- Script: [src/baselines/run_gold_b_cross_encoder.py](../../src/baselines/run_gold_b_cross_encoder.py)
- Rationale: Evaluate the pretrained BGE reranker on `gold_B` with top-k decoding only, without fine-tuning or `gold_B`-trained thresholds. This provides the `gold_B` counterpart of the `gold_A` no-training reranker comparison.
- Params:
  - `model_name`: `BAAI/bge-reranker-v2-m3`
  - `text_mode`: `concise`
  - `topks`: `1,2,3,5`
  - `decoding`: `top-k by reranker score`
- Metrics:
  - `best_topk`: `1`
  - `best_pair_f1`: `0.585635`
  - `best_pair_precision`: `0.598870`
  - `best_pair_recall`: `0.572973`
  - `best_record_exact_match_rate`: `0.435028`
- Outputs:
  - [outputs/baselines/gold_B_cross_encoder/oof_predictions.pkl](../../outputs/baselines/gold_B_cross_encoder/oof_predictions.pkl)
  - [outputs/baselines/gold_B_cross_encoder/topk_summary.pkl](../../outputs/baselines/gold_B_cross_encoder/topk_summary.pkl)
  - [outputs/baselines/gold_B_cross_encoder/topk_summary.csv](../../outputs/baselines/gold_B_cross_encoder/topk_summary.csv)
