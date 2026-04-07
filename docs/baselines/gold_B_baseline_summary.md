# gold_B baseline まとめ

## 目的

`gold_B` を使って、API に依存しない baseline を再構成し、`gold_A` に適用する前の比較基準を整理する。

前提は次の通り。

- 候補 synset は `sids_JA` と `sids_JMdict` のみを使う
- 評価では `EQUAL` のみを正例とする
- baseline 関連のスクリプトは `src/baselines/` にまとめる
- 実験順と結果は `docs/baselines/EXPERIMENT_LOG.md` に残す

## 今回の再実行順

1. pairwise lexical baseline
2. E5 ranking baseline
3. hybrid search
4. record-level decoder

`cross-encoder` の関連ファイルは残しているが、今回は再実行していない。

## 主なスクリプト

- `src/baselines/run_gold_b_baseline.py`
  - pairwise lexical baseline
- `src/baselines/run_gold_b_ranking.py`
  - E5 による ranking baseline
  - `../final` 由来の query / passage テンプレートも比較
- `src/baselines/run_gold_b_hybrid_search.py`
  - pairwise 特徴量と ranking 特徴量を組み合わせて best 構成を探索
- `src/baselines/run_gold_b_record_decoder.py`
  - best hybrid の pairwise score を record 単位に復号

## 再実行結果

| 段階 | 主な設定 | 主指標 | 値 |
|---|---|---:|---:|
| Pairwise baseline | logistic regression | pairwise `EQUAL` F1 | `0.545918` |
| Ranking baseline | `natural + ja_preferred + top1` | pairwise `EQUAL` F1 | `0.657459` |
| Hybrid best | `natural + ja_preferred_top2` | pairwise `EQUAL` F1 | `0.684466` |
| Record decoder | best hybrid から復号 | record exact match rate | `0.491525` |

## 補足

### Ranking baseline

`run_gold_b_ranking.py` を整理し直した上で再実行したところ、ranking 単体では

- `text_mode = natural`
- `candidate_mode = ja_preferred`
- `topk = 1`

が最良で、`pairwise EQUAL F1 = 0.657459` を再現できた。

この値は旧 summary に書かれていた ranking best と一致している。

### Hybrid baseline

旧 summary には `pairwise F1 = 0.682171` が書かれていたが、当時の実験順・設定が十分に保存されていなかったため、そのままの条件は復元できなかった。

今回の再構成では、

- ranking の複数テンプレート
- ranking feature subset の複数構成

を `run_gold_b_hybrid_search.py` で比較した。その結果、再現可能な best は

- `text_mode = natural`
- `feature_preset = ja_preferred_top2`

で、`pairwise F1 = 0.684466` だった。

つまり、旧 summary にあった `0.682171` そのものは今回の clean rerun では確認できていないが、それを上回る再現可能な設定は得られている。

best 構成は次に保存している。

- `outputs/baselines/gold_B_hybrid_best/config.json`
- `outputs/baselines/gold_B_hybrid_best/metrics.pkl`

### Record decoder

record-level decoder は `gold_B_hybrid_best` を入力にして再実行した。

結果は

- `pair_f1 = 0.661290`
- `record_macro_f1 = 0.616196`
- `record_exact_match_rate = 0.491525`

で、record 完全一致の観点では最終段として有効だった。

## 旧 summary との差分

旧 summary は `docs/baselines/gold_B_baseline_summary_legacy.md` に退避してある。

今回の clean rerun との差分は主に次の 2 点。

1. ranking best は旧 summary と同じ `0.657459` を再現できた
2. hybrid best は旧 summary の `0.682171` ではなく、再現可能な best として `0.684466` が得られた

したがって、今後は legacy summary をそのまま正とせず、`outputs/baselines/gold_B_hybrid_best/` の設定と `docs/baselines/EXPERIMENT_LOG.md` を正規の記録として扱う。

## 次に見る場所

- 実験順と主なパラメータ:
  - `docs/baselines/EXPERIMENT_LOG.md`
- ranking の比較結果:
  - `outputs/baselines/gold_B_ranking/metrics.pkl`
- hybrid の比較結果:
  - `outputs/baselines/gold_B_hybrid_search/summary.pkl`
- 現在の best 構成:
  - `outputs/baselines/gold_B_hybrid_best/config.json`
