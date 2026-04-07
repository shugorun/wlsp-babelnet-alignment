# Baselines

このディレクトリは、WLSP-BabelNet alignment に対する非 API ベースラインの説明と結果をまとめる場所です。

## 対象設定

- 候補 synset は `sids_JA` と `sids_JMdict` を中心に扱う
- 主評価は `EQUAL` の二値判定
- `gold_B` で比較し、必要に応じて `gold_A` へ適用する

## 実験の流れ

1. pairwise lexical baseline
2. E5 ranking baseline
3. hybrid search
4. record-level decoder

補助的に cross-encoder や外部モデル比較も行っていますが、主軸は上の 4 段です。

## 主なスクリプト

- `src/baselines/run_gold_b_baseline.py`
  - lexical 特徴量ベースの pairwise baseline
- `src/baselines/run_gold_b_ranking.py`
  - E5 を用いた ranking baseline
- `src/baselines/run_gold_b_hybrid_search.py`
  - lexical 特徴量と ranking 特徴量の組み合わせ探索
- `src/baselines/run_gold_b_record_decoder.py`
  - pairwise score を record 単位の出力へ変換
- `src/baselines/run_gold_a_hybrid_best.py`
  - `gold_B` best hybrid を `gold_A` に適用
- `src/baselines/run_gold_a_no_training_methods.py`
  - `gold_B` 学習なしの手法比較

## 現在採用している `gold_B` best

- Hybrid best
  - `outputs/baselines/gold_B_hybrid_best/config.json`
  - pairwise `EQUAL` F1 = `0.684466`
- Record decoder best
  - record exact match rate = `0.491525`

## `gold_A` での位置づけ

`gold_A` では、`gold_B` で選んだ best hybrid をそのまま適用した結果と、  
`gold_B` を学習に使わない手法群を比較しています。

- `gold_B` best hybrid をそのまま適用
  - pairwise `EQUAL` F1 = `0.531915`
- `gold_B` 学習なしの best
  - pretrained cross-encoder
  - pairwise `EQUAL` F1 = `0.551282`

この差分は、`gold_B` で最適だった設定がそのまま `gold_A` で最適になるとは限らないことを示しています。

## 再現性メモ

- 旧 summary にあった `0.682171` は legacy 値です。
- 現在の clean rerun で再現可能な hybrid best は `0.684466` です。
- 公開版では、再現可能な best を基準に説明しています。

## 文書

- 実験ログ: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- `gold_B` 要約: [gold_B_baseline_summary.md](gold_B_baseline_summary.md)
- 旧要約: [gold_B_baseline_summary_legacy.md](gold_B_baseline_summary_legacy.md)

## 公開版で残しているもの

公開版では、BabelNet 由来の重い特徴量 dump や中間生成物は含めず、主に次を残しています。

- 実験コード
- 実験ログ
- 再現可能な best 設定
- 軽量な metrics / summary
