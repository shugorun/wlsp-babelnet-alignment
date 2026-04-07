# WLSP-BabelNet Alignment

WLSP の分類語彙レコードと BabelNet synset の対応付けを行う研究用リポジトリです。  
主な関心は、限られた候補 synset 集合に対して、どの候補が `EQUAL` に当たるかを見分けることです。

このリポジトリでは、主に次の 2 系統を扱っています。

- 非 API ベースライン
  - lexical 特徴量
  - 埋め込み ranking
  - hybrid
  - record-level decoder
- API ベースの alignment 実験
  - 構造化出力を用いた candidate ごとのラベル付け

## このリポジトリで見せたいこと

- 候補生成と候補選別を分けたアラインメント設計
- lexical / semantic / ranking を組み合わせたベースライン構築
- `gold_B` を用いた比較実験と、`gold_A` への転移評価
- 実験ログ、設定、要約を分けた再現性重視の整理

## ディレクトリ案内

- `src/`
  - 実験コード本体
- `src/baselines/`
  - 非 API ベースライン
- `src/alignment/`
  - API alignment 実験
- `docs/baselines/`
  - ベースラインの説明、結果、実験ログ
- `outputs/baselines/`
  - 公開してよい軽量な設定・要約のみ

## 主要結果

`gold_B` 上の再現可能な best は次です。

- Hybrid best
  - pairwise `EQUAL` F1 = `0.684466`
- Record decoder best
  - record exact match rate = `0.491525`

`gold_A` では、`gold_B` 学習なしの比較も行っています。  
詳細は [docs/baselines/README.md](docs/baselines/README.md) を参照してください。

## 再現できる範囲

公開版では、次のレベルの再現を想定しています。

- 実験設計の理解
- ベースライン実装の読解
- 実験設定と主要結果の確認
- 各スクリプトの入出力関係の確認

一方で、次はそのままでは再現できません。

- BabelNet データ本体を必要とする処理
- BabelNet 由来の中間生成物をそのまま使う再実行
- API の生出力を含む完全な再実行

つまり、この公開版は「研究の構造と工夫が追えること」を優先していて、  
ライセンス上・公開方針上の制約があるデータ本体は含めていません。

## 詳細ドキュメント

- ベースライン: [docs/baselines/README.md](docs/baselines/README.md)
- API alignment: [docs/alignment/README.md](docs/alignment/README.md)
- データと公開方針: [data/README.md](data/README.md)
- 公開版 outputs の扱い: [outputs/README.md](outputs/README.md)

## 再現性と公開方針

この公開版リポジトリには、ライセンス上または公開方針上そのまま含めないものがあります。

- BabelNet データ本体
- BabelNet 由来の中間生成物
- API 実行の生出力
- 一部の raw / processed data

そのため、このリポジトリでは

- コード
- 軽量な設定ファイル
- 実験要約
- ログ文書

を中心に公開しています。

## ライセンス上の注意

- WLSP は CC BY-NC-SA 3.0 に従って利用しています。
- BabelNet は BabelNet license に従って利用しています。
- 公開版リポジトリには、BabelNet データ本体およびそのまま再配布にあたる生成物は含めていません。

ライセンスの詳細やデータの扱いは [data/README.md](data/README.md) にまとめています。
