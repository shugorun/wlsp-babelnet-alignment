from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.key import load_openai_config


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def list_input_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.json") if p.is_file()])


def extract_output_text(resp: Any) -> str:
    """Extract output text from one Responses API object."""
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text

    try:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            if not isinstance(item, dict):
                continue
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    parts.append(c["text"])
        s = "".join(parts).strip()
        if s:
            return s
    except Exception:
        pass

    raise RuntimeError("Could not extract output text from response.")


def validate(inp: Dict[str, Any], out: Dict[str, Any]) -> Tuple[bool, str]:
    rid_in = str(inp.get("record_id", ""))
    rid_out = str(out.get("record_id", ""))

    if rid_in and rid_out and rid_in != rid_out:
        return False, f"record_id mismatch: in={rid_in} out={rid_out}"

    cands = inp.get("candidates", [])
    labels = out.get("labels", [])

    if not isinstance(cands, list) or not isinstance(labels, list):
        return False, "candidates/labels must be lists"

    in_sids = [c.get("synset_id") for c in cands if isinstance(c, dict)]
    out_sids = [x.get("synset_id") for x in labels if isinstance(x, dict)]

    if len(in_sids) != len(out_sids):
        return False, f"count mismatch: candidates={len(in_sids)} labels={len(out_sids)}"

    if set(in_sids) != set(out_sids):
        return False, "synset_id set mismatch"

    allowed = {"EQUAL", "HYPERNYM", "HYPONYM", "NONE"}
    for x in labels:
        if not isinstance(x, dict):
            return False, "labels item is not an object"
        if x.get("label") not in allowed:
            return False, f"invalid label: {x.get('label')}"

    return True, "OK"


def call_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
    prompt_cache_key: str,
    max_retries: int,
    base_sleep: float,
) -> Any:
    """Call the Responses API with structured outputs."""
    user_text = json.dumps(user_payload, ensure_ascii=False)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                reasoning={"effort": "high"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "wlsp_babelnet_alignment",
                        "schema": schema,
                        "strict": True,
                    }
                },
                prompt_cache_key=prompt_cache_key,
            )
        except Exception as e:
            last_err = e
            sleep = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.2
            print(f"[WARN] attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(sleep)

    raise RuntimeError(f"API failed after {max_retries} retries: {last_err}")


ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-5.4-2026-03-05"
MAX_FILES = 0
FORCE = False
MAX_RETRIES = 5
BASE_SLEEP = 1.0
KEY_PATH = ".vscode/openai-key.json"
CACHE_KEY = "wlsp_babelnet_alignment_v1"
INPUT_DIR = ROOT / "data" / "interim" / "api_inputs" / "alignment" / "version_1" / "gold_A"
OUT_DIR = ROOT / "outputs" / "api_runs" / "alignment" / "gpt-5.4-2026-03-05" / "gold_A"
PROMPT_VERSION = "v1"
SCHEMA_VERSION = "v1"
PROMPT_DIR = ROOT / "src" / "alignment" / "prompts"
SCHEMA_DIR = ROOT / "src" / "alignment" / "schemas"


def load_prompt(version: str) -> str:
    return (PROMPT_DIR / f"alignment_prompt_{version}.txt").read_text(encoding="utf-8")


def load_schema(version: str) -> Dict[str, Any]:
    return json.loads((SCHEMA_DIR / f"alignment_schema_{version}.json").read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_openai_config(KEY_PATH)
    client = OpenAI(api_key=cfg["openai_api_key"])
    system_prompt = load_prompt(PROMPT_VERSION)
    output_schema = load_schema(SCHEMA_VERSION)

    input_dir = INPUT_DIR
    out_dir = OUT_DIR
    resp_dir = out_dir / "responses"
    parsed_dir = out_dir / "parsed"
    err_dir = out_dir / "errors"

    files = list_input_files(input_dir)
    if MAX_FILES and MAX_FILES > 0:
        files = files[:MAX_FILES]

    print(f"[INFO] model={MODEL}")
    print(f"[INFO] cache_key={CACHE_KEY}")
    print(f"[INFO] prompt_version={PROMPT_VERSION}")
    print(f"[INFO] schema_version={SCHEMA_VERSION}")
    print(f"[INFO] files={len(files)} input_dir={input_dir}")
    print(f"[INFO] out_dir={out_dir}")

    ok = 0
    ng = 0

    for p in files:
        base = p.stem
        out_resp_path = resp_dir / f"{base}.json"
        out_parsed_path = parsed_dir / f"{base}.json"

        if out_parsed_path.exists() and not FORCE:
            print(f"[SKIP] {out_parsed_path.name}")
            ok += 1
            continue

        inp = load_json(p)

        try:
            resp = call_with_retry(
                client=client,
                model=MODEL,
                system_prompt=system_prompt,
                user_payload=inp,
                schema=output_schema,
                prompt_cache_key=CACHE_KEY,
                max_retries=MAX_RETRIES,
                base_sleep=BASE_SLEEP,
            )

            try:
                raw = resp.model_dump()
            except Exception:
                raw = {"_repr": repr(resp)}
            dump_json(out_resp_path, raw)

            out_text = extract_output_text(resp)
            out_obj = json.loads(out_text)

            valid, msg = validate(inp, out_obj)
            if not valid:
                raise RuntimeError(f"validation failed: {msg}")

            dump_json(out_parsed_path, out_obj)
            print(f"[OK] {p.name} -> {out_parsed_path.name}")
            ok += 1

        except Exception as e:
            ng += 1
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / f"{base}.txt").write_text(str(e), encoding="utf-8")
            print(f"[NG] {p.name}: {type(e).__name__}: {e}")

    print(f"[DONE] ok={ok} ng={ng} out_dir={out_dir}")


if __name__ == "__main__":
    main()
