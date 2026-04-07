# Outputs Policy

この公開版では、`outputs/` を「すべての生成物を置く場所」としては扱っていません。  
公開してよい軽量な成果物だけを残し、それ以外の生成物は Git 管理から外しています。

## 残しているもの

- `outputs/baselines/gold_B_hybrid_best/config.json`
  - `gold_B` 上の再現可能な best hybrid 設定
- `outputs/baselines/gold_B_hybrid_best/metrics.pkl`
  - 上記設定の要約指標
- `outputs/baselines/gold_B_hybrid_search/summary.pkl`
  - hybrid 設定探索の要約

## 含めていないもの

- BabelNet 由来の重い特徴量 dump
- pairwise 予測の全件保存
- API 実行の生レスポンス
- ranking の全文テキストや candidate 内容を含む中間生成物

## 方針

公開版で重視しているのは、次の 2 点です。

1. どの手法を比較し、どの設定が best だったかが追えること
2. ライセンス上問題のあるデータや派生物を含めないこと

そのため、`outputs/` は「研究メモの完全アーカイブ」ではなく、  
公開向けに最小限の設定・要約だけを残す構成にしています。
