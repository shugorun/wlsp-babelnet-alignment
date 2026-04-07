# Alignment Prompt Registry

## Prompt

- `alignment_prompt_v1.txt`
  - `src/alignment/run_alignment.py` で使っていた元の prompt をそのまま保存した版
- `alignment_prompt_v2.txt`
  - 判定手順を段階化し、`NONE` を優先する短めの版

## Schema

- `alignment_schema_v1.json`
  - 元の schema をそのまま保存した版
- `alignment_schema_v2.json`
  - 現状は `v1` と同一構造。prompt 実験と対応を取りやすくするため版を分けている
