# Outputs Policy

公開してよい軽量な成果物だけを残し、それ以外の生成物は Git 管理から外しています。

---

## 残しているもの

- [outputs/baselines/gold_B_hybrid_best/config.json](baselines/gold_B_hybrid_best/config.json)
  - Gold B 上の再現可能な best hybrid 設定
- [outputs/baselines/gold_B_hybrid_best/metrics.pkl](baselines/gold_B_hybrid_best/metrics.pkl)
  - 上記設定の要約指標
- [outputs/baselines/gold_B_hybrid_search/summary.pkl](baselines/gold_B_hybrid_search/summary.pkl)
  - hybrid 設定探索の要約
- [outputs/baselines/gold_B_cross_encoder/topk_summary.csv](baselines/gold_B_cross_encoder/topk_summary.csv)
  - 事前学習済み BGE reranker の Gold B no-training top-k 評価要約
- [outputs/api_runs/term_expansion/version_1/gold_A/parsed/](api_runs/term_expansion/version_1/gold_A/parsed/)
- [outputs/api_runs/term_expansion/version_1/gold_B/parsed/](api_runs/term_expansion/version_1/gold_B/parsed/)
  - 分類語彙表を入力として LLM が生成した検索語拡張結果
  - 分類語彙表の CC BY-NC-SA 3.0 に従い、非営利目的の研究公開として残しています

---

## 含めていないもの

- BabelNet 由来の重い特徴量 dump
- pairwise 予測の全件保存
- API 実行の生レスポンス
- ranking の全文テキストや候補内容を含む中間生成物

---

## 方針

公開版の方針は、次の 2 点です。

1. どの手法を比較し、どの設定が best だったかが追えること
2. ライセンス上問題のあるデータや派生物を含めないこと

そのため、[outputs/](./) は「研究メモの完全アーカイブ」ではなく、公開向けに最小限の設定・要約だけを残す構成にしています。
