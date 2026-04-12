# Prompt / Schema バージョン管理

このファイルは、アラインメント実験で使用した prompt と schema のバージョン履歴を記録します。  
詳細な設計説明は [docs/alignment/](../../docs/alignment/) 以下のドキュメントを参照してください。

---

## バージョン一覧

| prompt | schema | 組み合わせ名 | 状態 | 説明 |
|---|---|---|---|---|
| v1 | v1 | `{model}_high_v1_v1` | 実験済み（結果あり） | 詳細ルール・few-shot 例付きの保守的プロンプト |
| v2 | v1 | `{model}_high_v2_v1` | 未実行（実施予定） | v1 を step 形式に整理・few-shot 例なし |

---

## Prompt

### v1

- ファイル: [prompts/alignment_prompt_v1.txt](prompts/alignment_prompt_v1.txt)
- 設計説明: [docs/alignment/prompts/v1.md](../../docs/alignment/prompts/v1.md)
- 特徴: few-shot 例あり、判定ルールを細かく列挙、`NONE` を強く優先

### v2

- ファイル: [prompts/alignment_prompt_v2.txt](prompts/alignment_prompt_v2.txt)
- 設計説明: [docs/alignment/prompts/v2.md](../../docs/alignment/prompts/v2.md)
- 特徴: few-shot 例なし、判定手順を step 形式で整理、v1 と同じ保守的方針を維持

---

## Schema

### v1

- ファイル: [schemas/alignment_schema_v1.json](schemas/alignment_schema_v1.json)
- 設計説明: [docs/alignment/schemas/v1.md](../../docs/alignment/schemas/v1.md)
- 構造: `record_id` + `labels[]`（`synset_id` / `reasoning` / `label`）
- `label` は `EQUAL / HYPERNYM / HYPONYM / NONE` の enum
- 余分なフィールドを許可しない（`additionalProperties: false`）

---

## 命名規則

実行結果ディレクトリは次の形式で命名します。

```
{model}_{reasoning_effort}_{prompt_version}_{schema_version}
```

例: `gpt-5.2-2025-12-11_high_v1_v1`
