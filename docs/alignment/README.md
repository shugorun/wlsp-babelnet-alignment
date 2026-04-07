# API Alignment

このディレクトリが対象とするのは、API を使った alignment 実験です。

## 目的

WLSP record と BabelNet candidate synset の対応関係を、構造化出力で `EQUAL / HYPERNYM / HYPONYM / NONE` に分類します。

## 主要ファイル

- `src/alignment/alignment_inputs.py`
  - API に渡す入力 JSON を生成
- `src/alignment/run_alignment.py`
  - prompt / schema を読み、Responses API を実行
- `src/alignment/prompts/`
  - prompt version を管理
- `src/alignment/schemas/`
  - schema version を管理
- `src/alignment/prompt_registry.md`
  - prompt / schema の変更履歴メモ

## 方針

- prompt と schema はコードに直書きせず、版管理されたファイルとして分離
- `gold_B` を使って prompt 改善の比較を行い、その後に `gold_A` へ適用
- 誤り分析をもとに、few-shot や判定手順の改善を行う

## 注意

公開版リポジトリには API の生レスポンスや candidate 内容を含む中間生成物は含めません。  
必要な場合は、各自の環境でデータと API key を設定して再生成する前提です。
