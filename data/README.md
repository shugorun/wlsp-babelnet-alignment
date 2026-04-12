# データ構造とライセンス方針

このリポジトリでは、データと生成物をライセンスと再配布条件に合わせて制限しています。ここでは公開版で確認できるデータの構造と、含めていないものの理由をまとめます。

---

## データセット概要

本研究では 2 種類の正解データセットを使用しています。

| データセット | n_records | n_pairs | 用途 |
|---|---:|---:|---|
| Gold B | 177 | 2,654 | ベースライン開発・設定選択 |
| Gold A | 86 | 1,003 | 転移評価・最終評価 |

- **`n_records`**: 評価対象の分類語彙表レコード数
- **`n_pairs`**: 評価対象の（レコード, 候補 synset）ペア数
- **Gold B** は設定探索や比較に使い、**Gold A** は Gold B で決めた設定をそのまま適用して汎化性能を確認するために使います

---

## ディレクトリ構造

```
data/
├── gold/                            # 正解データ（公開版に含む）
│   ├── gold_A.pkl                   # Gold A 正解ラベル
│   ├── gold_A_records.pkl           # Gold A レコード（候補 synset 付き）
│   ├── gold_A_records.parquet       # 同上（parquet 形式）
│   ├── gold_B.pkl                   # Gold B 正解ラベル
│   ├── gold_B_records.pkl           # Gold B レコード（候補 synset 付き）
│   └── gold_B_records.parquet       # 同上（parquet 形式）
│
├── interim/                         # API 入力用の中間生成物
│   └── api_inputs/
│       ├── alignment/version_1/     # LLM アラインメント入力 JSON
│       │   ├── gold_A/              # Gold A 分の入力チャンク
│       │   └── gold_B/              # Gold B 分の入力チャンク
│       └── term_expansion/version_1/
│           ├── gold_A/              # Gold A 分の term expansion 入力
│           └── gold_B/              # Gold B 分の term expansion 入力
│
└── processed/                       # BabelNet 前処理済みデータ（非公開）
    ├── wlsp.pkl                     # 分類語彙表の全レコード
    └── babelnet_.pkl                # BabelNet synset 情報（BabelNet ライセンスのため非公開）
```

---

## 各ファイルの内容

### gold/

| ファイル | 内容 |
|---|---|
| `gold_A_records.pkl` | Gold A の各レコード（見出し語・カテゴリパス・候補 synset リストなど） |
| `gold_B_records.pkl` | Gold B の各レコード（同上） |
| `gold_A.pkl` | Gold A の正解ラベル（レコード ID × synset ID → `EQUAL / HYPERNYM / HYPONYM / NONE`） |
| `gold_B.pkl` | Gold B の正解ラベル（同上） |
| `*.parquet` | 対応する `.pkl` と同内容の parquet 形式（pandas での読み込み用） |

### interim/api_inputs/

スクリプトが API に渡す入力 JSON を生成した中間ファイルです。  
`alignment_inputs.py` や `term_expansion_inputs.py` が出力します。

### processed/

BabelNet の synset 情報（lemma・gloss・上位語など）を前処理したファイルです。  
BabelNet ライセンスの制約により公開版には含めていません。各自で BabelNet データを取得・配置し、処理スクリプトを実行して生成する必要があります。

---

## 公開版で確認できるもの

- コード（`src/` 以下の全スクリプト）
- 正解データ（`data/gold/`）
- API 入力の中間生成物（`data/interim/api_inputs/`）
- term expansion の parsed 出力（`outputs/api_runs/term_expansion/`）
- ベースライン設定・評価要約（`outputs/baselines/`）
- 実験ドキュメント（`docs/`）

## 含めていないもの

| 種別 | 理由 |
|---|---|
| BabelNet データ本体 | BabelNet Non-Commercial License による制約 |
| `data/processed/babelnet_.pkl` | BabelNet 由来の派生物 |
| API の生レスポンス | データ量・ライセンス上の理由 |
| ranking の全文テキストや candidate 内容を含む中間生成物 | BabelNet 由来のデータを含むため |

---

## ライセンス

各データの利用にあたっては、以下のライセンスを確認したうえで、必要なデータを各自で取得・配置してください。

| データ | ライセンス |
|---|---|
| WLSP（分類語彙表） | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) |
| BabelNet | [BabelNet Non-Commercial License](https://babelnet.org/license) |
