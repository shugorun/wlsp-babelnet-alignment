# ベースライン実験

<p align="left">
  <img alt="Scope: Baselines" src="https://img.shields.io/badge/scope-baselines-555555">
  <img alt="No API" src="https://img.shields.io/badge/API-not%20used-2f6f9f">
  <img alt="Evaluation: EQUAL F1" src="https://img.shields.io/badge/evaluation-EQUAL%20F1-2f6f9f">
  <img alt="Models: E5 and BGE" src="https://img.shields.io/badge/models-E5%20%7C%20BGE-555555">
</p>

このディレクトリは、WLSP-BabelNet アラインメントに対する **API を使わないベースライン実験** の説明・結果・実験ログをまとめる場所です。

ベースラインの目的は、LLM アラインメントの比較基準を作ることです。  
`EQUAL` は、**分類語彙表レコードと候補 synset が同じ語義に対応している**ことを表します。

---

## 評価設定

主指標は pairwise F1 です。分類語彙表レコードと候補 synset の 1 ペアを 1 件として扱い、`EQUAL` を正しく判定できるかを評価します。

- **Gold B**: 手法の比較と設定選択に使用（n_records = 177、n_pairs = 2,654）
- **Gold A**: Gold B で選んだ設定の転移評価に使用（n_records = 86、n_pairs = 1,003）

「学習なし」は、本実験の正解データで追加学習しないことを意味します。

---

## 手法

| 手法 | 内容 |
|---|---|
| Pairwise lexical | 文字列一致や候補由来などの特徴量から `EQUAL` を判定 |
| E5 ranking | 多言語埋め込み類似度で候補を順位付け |
| BGE reranker | 事前学習済み cross-encoder で候補を順位付け |
| Hybrid | lexical 特徴量と E5 ranking スコアを組み合わせて `EQUAL` を判定 |
| Record decoder | pairwise スコアから record 単位の synset 集合を復元 |

---

## 学習あり手法

Pairwise lexical と Hybrid は、Gold B 上で GroupKFold により評価しました。  
同じレコードの候補ペアが train / test にまたがらないように分割し、train 側で LogisticRegression を学習します。閾値も train 側で決め、test 側に適用しています。

---

## 評価結果

| 評価データ | 手法 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Gold B | Pairwise lexical | 0.5169 | 0.5784 | 0.5459 |
| Gold B | E5 ranking | 0.6723 | 0.6432 | 0.6575 |
| Gold B | BGE reranker | 0.5989 | 0.5730 | 0.5856 |
| Gold B | Hybrid best | 0.6211 | 0.7622 | **0.6845** |
| Gold A | Gold B hybrid 転移 | 0.4237 | 0.7143 | 0.5319 |
| Gold A | BGE reranker | 0.5000 | 0.6143 | 0.5513 |

- Gold B では Hybrid best が最も高い F1 を示しました。
- Gold A では、Gold B で決めた Hybrid best の設定とモデルをそのまま適用しました。

---

## Record 単位の評価

`record exact match` は、1 つの分類語彙表レコードについて、正解の `EQUAL` synset 集合と予測した `EQUAL` synset 集合が完全に一致した割合です。1 つでも余分な synset を選ぶ、または必要な synset を落とすと、そのレコードは不一致になります。  
`record macro F1` は、この synset 集合の precision / recall / F1 をレコードごとに計算し、平均した値です。

| 評価データ | 手法 | pairwise F1 | record macro F1 | record exact match |
|---|---|---:|---:|---:|
| Gold B | Hybrid best | 0.6845 | — | 0.4746 |
| Gold B | Record decoder | 0.6613 | 0.6162 | 0.4915 |
| Gold A | Gold B hybrid 転移 | 0.5319 | 0.4570 | 0.3023 |

Record decoder は pairwise F1 を少し下げる一方で、record exact match を改善しています。

---

## 実験ログ

- 英語版（canonical）: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- 日本語版: [EXPERIMENT_LOG_ja.md](EXPERIMENT_LOG_ja.md)
- Hybrid best の設定: [outputs/baselines/gold_B_hybrid_best/config.json](../../outputs/baselines/gold_B_hybrid_best/config.json)

---

## 主要ファイル

| スクリプト | 役割 |
|---|---|
| [src/baselines/run_gold_b_baseline.py](../../src/baselines/run_gold_b_baseline.py) | Gold B 上で lexical ベースラインを評価 |
| [src/baselines/run_gold_b_ranking.py](../../src/baselines/run_gold_b_ranking.py) | E5 ranking ベースラインを評価 |
| [src/baselines/run_gold_b_cross_encoder.py](../../src/baselines/run_gold_b_cross_encoder.py) | BGE reranker を評価 |
| [src/baselines/run_gold_b_hybrid_search.py](../../src/baselines/run_gold_b_hybrid_search.py) | hybrid の特徴量構成を探索 |
| [src/baselines/run_gold_b_record_decoder.py](../../src/baselines/run_gold_b_record_decoder.py) | record 単位の出力を復元 |
| [src/baselines/run_gold_a_hybrid_best.py](../../src/baselines/run_gold_a_hybrid_best.py) | Gold B で選んだ Hybrid best を Gold A に転移・評価 |
| [src/baselines/run_gold_a_no_training_methods.py](../../src/baselines/run_gold_a_no_training_methods.py) | 学習なし手法を Gold A で比較 |
