# src/common/key.py
from __future__ import annotations

import json
from pathlib import Path


def load_openai_config(key_path: str | Path = ".vscode/openai-key.json") -> dict:
    p = Path(key_path)
    if not p.is_absolute():
        # 実行カレント基準。プロジェクトルートから実行する運用を推奨。
        p = Path.cwd() / p

    if not p.exists():
        raise FileNotFoundError(f"Key file not found: {p}")

    cfg = json.loads(p.read_text(encoding="utf-8"))
    if "openai_api_key" not in cfg or not str(cfg["openai_api_key"]).strip():
        raise ValueError("openai_api_key is missing/empty in key file.")
    return cfg
