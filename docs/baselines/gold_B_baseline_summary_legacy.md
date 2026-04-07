# gold_B ベースライン実験まとめ

## 概要

`gold_B` を用いて、API に依存しない機械的手法によるベースラインを構築した。

今回の前提は以下の通りである。

- 候補 synset は `sids_JA ∪ sids_JMdict` に限定する
- 評価では `EQUAL` のみを正例とする
- `NONE`, `HYPONYM`, `HYPERNYM` は負例として扱う

実験は以下の 4 段階で進めた。

1. 語彙・文字列ベースの pairwise baseline
2. 埋め込みによる候補内 ranking baseline
3. pairwise 特徴量と ranking 特徴量を組み合わせた hybrid baseline
4. record 単位で出力を制御する decoder


## 使用データ

実験で用いたファイルは以下の pickle のみである。

- `data/gold/gold_B_records.pkl`
- `data/gold/gold_B.pkl`
- `data/processed/babelnet_.pkl`


## 実装したスクリプト

### 1. Pairwise baseline

ファイル:

- `src/run_gold_b_baseline.py`

内容:

- `record_id x synset_id` 単位の特徴量を作成
- 候補流入元、文字列一致、gloss/category の整合性などを特徴量化
- weighted scoring と logistic regression を評価

主な出力:

- `outputs/baselines/gold_B/pair_features.pkl`
- `outputs/baselines/gold_B/oof_predictions.pkl`
- `outputs/baselines/gold_B/metrics.pkl`
- `outputs/baselines/gold_B/logreg_model.pkl`


### 2. Ranking baseline

ファイル:

- `src/run_gold_b_ranking.py`

内容:

- `../final` で用いていた ranking の考え方をこのリポジトリ側で再実装
- WLSP record ごとに query 文を作成
- BabelNet synset ごとに passage 文を作成
- multilingual E5 により埋め込みを計算
- 各 record の固定候補集合内で synset を順位付け

主な出力:

- `outputs/baselines/gold_B_ranking/rankings.pkl`
- `outputs/baselines/gold_B_ranking/metrics.pkl`


### 3. Hybrid baseline

ファイル:

- `src/run_gold_b_hybrid.py`

内容:

- pairwise 特徴量と ranking 特徴量を結合
- logistic regression を学習
- `EQUAL` 対それ以外の pairwise 分類として評価

主な出力:

- `outputs/baselines/gold_B_hybrid/pair_features_hybrid.pkl`
- `outputs/baselines/gold_B_hybrid/oof_predictions.pkl`
- `outputs/baselines/gold_B_hybrid/metrics.pkl`
- `outputs/baselines/gold_B_hybrid/logreg_model.pkl`


### 4. Record-level decoder

ファイル:

- `src/run_gold_b_record_decoder.py`

内容:

- hybrid の pairwise score を入力とする
- record ごとに何件の synset を返すかを決定する
- 例えば以下のような単純なルールを探索する
  - 最大何件返すか
  - 最低 score
  - 1 位候補に対する score 比

主な出力:

- `outputs/baselines/gold_B_record_decoder/oof_predictions.pkl`
- `outputs/baselines/gold_B_record_decoder/metrics.pkl`
- `outputs/baselines/gold_B_record_decoder/logreg_model.pkl`


## ベスト結果

探索中に得られた各系統の最良結果は以下の通りである。

| 手法 | 主な評価指標 | 最良結果 |
|---|---:|---:|
| Pairwise baseline | pairwise `EQUAL` F1 | `0.545918` |
| Ranking baseline | pairwise `EQUAL` F1 | `0.657459` |
| Hybrid baseline | pairwise `EQUAL` F1 | `0.682171` |
| Record decoder | record exact match rate | `0.463277` |


## 各手法の結果

### 1. Pairwise baseline

最良結果:

- logistic regression
- pairwise `EQUAL` F1 = `0.545918`

解釈:

- 候補流入元や文字列一致だけでも一定の性能は出る
- ただし embedding による ranking よりは明確に弱い


### 2. Ranking baseline

最良結果:

- candidate mode: `ja_preferred`
- 出力方式: `top1`
- pairwise `EQUAL` F1 = `0.657459`

解釈:

- 固定候補集合の内部で順位付けするだけでもかなり強い
- 初期の lexical baseline を大きく上回った


### 3. Hybrid baseline

最良結果:

- pair precision = `0.653465`
- pair recall = `0.713514`
- pair F1 = `0.682171`
- record macro F1 = `0.584746`
- record exact match rate = `0.446328`

解釈:

- pairwise `EQUAL` 判定では最も強かった
- lexical 特徴量と ranking 特徴量は相補的だった


### 4. Record-level decoder

exact match 重視での最良結果:

- pair F1 = `0.650667`
- record macro F1 = `0.600000`
- record exact match rate = `0.463277`

解釈:

- exact match を重視すると、plain hybrid より record exact match rate は改善した
- その代わり pairwise F1 は少し下がった


## 試したが改善しなかったこと

### 1. Ranking 用生成文の変更

試した内容:

- concise / natural の複数テンプレート
- `division/category/subcategory/class` の強調
- `synonyms` を `related terms` 扱いに変更
- BabelNet 側を definition-first に変更

結果:

- これらの変更では earlier best の ranking 結果を上回れなかった
- ranking の最良値は従来の `ja_preferred + top1` のままだった


### 2. Hybrid への追加特徴量

試した内容:

- top-1 との差分 score
- top-1 に対する score ratio

結果:

- hybrid の性能改善にはつながらなかった
- exact-match-oriented decoder に対しても改善は見られなかった


## 採用候補

### pairwise 性能を重視する場合

採用候補:

- best hybrid baseline

参照値:

- pairwise `EQUAL` F1 = `0.682171`


### record 単位の exact match を重視する場合

採用候補:

- exact-match-oriented record decoder

参照値:

- record exact match rate = `0.463277`


## 現時点での結論

`gold_B` 上での探索としては、すでに以下の流れを一通り示せている。

1. lexical baseline
2. embedding ranking
3. hybrid reranking
4. record-level decoding

そのため、次にやるべきことは `gold_B` 上でさらに細かく調整することではなく、

- 採用する checkpoint を固定する
- それを `gold_A` に適用する
- 予測結果と評価結果を保存する

ことである。


## 補足

探索の後半では、いくつかの実験が `outputs/baselines/...` 以下の最新ファイルを上書きしている。

したがって、このまとめでは

- 探索中に観測された最良結果
- 現在ディスク上にある最新ファイル

を区別している。

報告や比較で重要なのは、上に記した「最良結果」の方である。
