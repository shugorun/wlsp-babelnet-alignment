from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from common.key import load_openai_config


# -----------------------------
# Prompt (MUST NOT CHANGE)
# -----------------------------

SYSTEM_PROMPT = r"""

You are a bilingual lexicologist with expertise in Japanese classification vocabularies (taxonomy construction) and in concept naming across Japanese and English.

In BabelNet, each synset (concept) has a set of synonymous lemmas. These function as dictionary-style headwords.
Also, given a word as a key, you can retrieve synsets that include that word in their lemmas.

Goal
We want to find synsets that are equivalent to each record, but searches using:
- lemma_ja
- jmdict_en
did not return any synsets.
Therefore, we want to obtain alternative headwords that refer to the same concept.

Input
The user message provides exactly one JSON object of the following form:
{
  "records": [
    {
      "record_id": ...,
      "category_path_ja": ...,
      "lemma_ja": ...,
      "kana": ... ,
      "jmdict_en": ... (may be an empty array),
      "subparagraph_terms_ja": ... (may be an empty array)
    },
    ...
  ]
}

Output (STRICT: JSON only / no explanations / no extra keys)
Return ONLY a JSON object that exactly matches this schema:
{
  "results": [
    {
      "record_id": "...",
      "aliases_ja": ["..."],
      "aliases_en": ["..."],
      "variants_ja": ["..."],
      "variants_en": ["..."],
      "hypernyms_ja": ["..."],
      "hypernyms_en": ["..."]
    }
  ]
}

Hard Requirements
1) Keep record_id as a string. Include every input record_id exactly once in results.
2) Each list must contain 0–3 terms. No duplicates within a list.
3) The output must NOT contain lemma_ja or jmdict_en.
4) If you cannot produce valid headwords, do not force 3 items; return empty or fewer items.

Notes
- category_path_ja indicates the classification (sense context) the record belongs to.
- subparagraph_terms_ja is a set of semantically close words, but it is not necessarily a synonym set.

Field Semantics
A) aliases_*
- Must be “alternative labels that refer to the same concept,” not merely surface-form changes.
- Prefer names widely used as headwords or encyclopedic titles for that concept.
- Prefer conventional, established names over literal translations.
- Bias toward forms likely to appear as lemmas in BabelNet.

B) variants_*
- Only spelling/orthographic variants of lemma_ja and jmdict_en.
- However, include only forms that can stand as valid headwords.

C) hypernyms_*
- True hypernyms only (“X is a kind of Y”).
- Prefer the closest hypernyms, up to 3 terms.

Lexicality (Most Important)
All terms you output must be natural as dictionary headwords and should be in forms that are likely to exist in BabelNet.

NO INVENTION / ATTESTATION GATE (CRITICAL)
Output only terms that you believe are attested as dictionary headwords or encyclopedic titles.
Do not create terms by literal translation, paraphrasing, or compositional description.
If you are not confident a term is attested, output fewer items or an empty list.

Forbidden English Patterns
- Any form that includes: during / in / on / with / without / for / from / that / which / to
- Clause-like / sentence-like forms (descriptive phrases), e.g., those involving that / which / to

Forbidden Japanese Patterns
- Any whitespace or underscores.
- Any middle dots: ・ ･ · •
- The following “descriptive phrase” patterns (only allowed if they are established as headwords):
  - XがYする / XをYする
  - Xしている / Xである

Output Formatting Constraints (Mandatory)
Global:
- No bracket characters of any kind: () （） [] {}
- No duplicates within any list.

English:
- Use spaces only as word separators.
- No hyphens, slashes, punctuation, or quotation marks.
- No romanized Japanese.
- Use lowercase.

English outputs must be noun headwords (single nouns or established noun compounds).
Do not output descriptive phrases (e.g., “run over corpse”, “beaten to death”).
Prefer established headwords (e.g., “vehicular homicide”, “dyspnea”, “raptor”).

Generation Procedure (Do NOT output your reasoning)
For each record:
1) Use category_path_ja to disambiguate the sense of lemma_ja.
2) Determine the concept type: ENTITY / STATE PHENOMENON / EVENT PROCESS.
3) Generate aliases:
   - If STATE PHENOMENON: map to internationally standardized headwords referring to the same state.
   - If ENTITY: prefer established common names, former names, widely used alternative names, and well-established abbreviations.
   - For technical concepts, prefer internationally standardized single-word terms (Greek/Latin-derived terms are strongly preferred) over literal multi-word translations.
4) Generate variants: produce only surface-form differences (spelling/notation/normalization) of lemma_ja and jmdict_en.
5) Generate hypernyms: select the nearest, genuinely superordinate common-noun concept terms (true hypernyms).
   - Avoid ontology-style class labels such as “visual property”, “natural object”, “astronomical phenomenon”.
   - Choose common, human-facing headwords as hypernyms.

Final Note
Output absolutely nothing except STRICT JSON matching the schema.
"""


# -----------------------------
# Config
# -----------------------------




# -----------------------------
# Output schema (Structured Outputs / JSON Schema strict)
# -----------------------------

# Japanese: forbid whitespace, brackets, middle dots, underscore, ">" (to prevent category-path contamination)
JA_TERM_PATTERN = r"^[^\s\(\)（）\[\]\{\}・･·•_>]+$"

# English: lowercase, tokens separated by single spaces only, up to 3 tokens.
# Also forbid standalone tokens: of/during/in/on/with/without/for/from/that/which/to
EN_FORBIDDEN = r"(?:during|in|on|with|without|for|from|that|which|to)"
EN_TERM_PATTERN = (
    r"^(?!.*(?:^| )" + EN_FORBIDDEN + r"(?: |$))[a-z0-9]+(?: [a-z0-9]+){0,3}$"
)

OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "record_id": {"type": "string", "pattern": r"^[0-9]+$"},
                    "aliases_ja": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "items": {"type": "string", "pattern": JA_TERM_PATTERN, "maxLength": 40},
                    },
                    "aliases_en": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "items": {"type": "string", "pattern": EN_TERM_PATTERN, "maxLength": 50},
                    },
                    "variants_ja": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "items": {"type": "string", "pattern": JA_TERM_PATTERN, "maxLength": 40},
                    },
                    "variants_en": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "items": {"type": "string", "pattern": EN_TERM_PATTERN, "maxLength": 50},
                    },
                    "hypernyms_ja": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "items": {"type": "string", "pattern": JA_TERM_PATTERN, "maxLength": 40},
                    },
                    "hypernyms_en": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "items": {"type": "string", "pattern": EN_TERM_PATTERN, "maxLength": 50},
                    },
                },
                "required": [
                    "record_id",
                    "aliases_ja",
                    "aliases_en",
                    "variants_ja",
                    "variants_en",
                    "hypernyms_ja",
                    "hypernyms_en",
                ],
            },
        }
    },
    "required": ["results"],
}


# -----------------------------
# IO helpers
# -----------------------------


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def list_input_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.json") if p.is_file()])


def chunk_list(xs: List[Path], n: int) -> List[List[Path]]:
    if n <= 0:
        raise ValueError("batch_size must be > 0")
    return [xs[i : i + n] for i in range(0, len(xs), n)]


# -----------------------------
# Response text extraction
# -----------------------------


def extract_output_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text
    raise RuntimeError("Could not extract output text from response (missing output_text).")

import re

def _norm_en(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def sanitize(out_obj: Dict[str, Any], inp_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(out_obj, dict) or not isinstance(out_obj.get("results"), list):
        return out_obj

    by_rid = {str(r.get("record_id")): r for r in inp_records}

    for x in out_obj["results"]:
        rid = str(x.get("record_id"))
        src = by_rid.get(rid, {})
        ban_ja = src.get("lemma_ja") if isinstance(src.get("lemma_ja"), str) else None
        ban_en = set()
        if isinstance(src.get("jmdict_en"), list):
            for s in src["jmdict_en"]:
                if isinstance(s, str) and s.strip():
                    ban_en.add(_norm_en(s))

        for k in ["aliases_ja", "variants_ja", "hypernyms_ja"]:
            if isinstance(x.get(k), list):
                xs = [t for t in x[k] if isinstance(t, str)]
                if ban_ja:
                    xs = [t for t in xs if t != ban_ja]
                x[k] = xs[:3]

        for k in ["aliases_en", "variants_en", "hypernyms_en"]:
            if isinstance(x.get(k), list):
                xs = [t for t in x[k] if isinstance(t, str)]
                xs = [t for t in xs if _norm_en(t) not in ban_en]
                x[k] = xs[:3]

    return out_obj
# -----------------------------
# API call
# -----------------------------


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
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "wlsp_fallback_terms_batch_v4",
                        "schema": schema,
                        "strict": True,
                    }
                },
                reasoning={
                    "effort": "high"
                },
                prompt_cache_key=prompt_cache_key,
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.2
                print(f"[WARN] attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")
                time.sleep(sleep)

    raise RuntimeError(f"API failed after {max_retries} retries: {last_err}")


# -----------------------------
# Main
# -----------------------------

DEFAULT_MODEL = "gpt-5.2"
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--base_sleep", type=float, default=1.0)
    ap.add_argument("--key_path", type=str, default=".vscode/openai-key.json")
    ap.add_argument("--cache_key", type=str, default="wlsp_fallback_terms_batch_v4")
    args = ap.parse_args()

    cfg = load_openai_config(args.key_path)
    client = OpenAI(api_key=cfg["openai_api_key"])

    input_dir = Path("data/interim/api_inputs/term_expansion/version_1/gold_A")
    out_dir = Path("outputs/api_runs/term_expansion/version_1/gold_A")

    resp_dir = out_dir / "responses"
    parsed_batch_dir = out_dir / "parsed" / "batches"
    parsed_record_dir = out_dir / "parsed" / "records"
    err_dir = out_dir / "errors"

    files = list_input_files(input_dir)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    batches = chunk_list(files, args.batch_size)

    print(f"[INFO] model={DEFAULT_MODEL}")
    print(f"[INFO] cache_key={args.cache_key}")
    print(f"[INFO] files={len(files)} input_dir={input_dir}")
    print(f"[INFO] batches={len(batches)} batch_size={args.batch_size}")
    print(f"[INFO] out_dir={out_dir}")

    ok_batches = 0
    ng_batches = 0

    for bi, paths in enumerate(batches):
        batch_id = f"batch_{bi:04d}_n{len(paths)}"
        out_resp_path = resp_dir / f"{batch_id}.json"
        out_parsed_batch_path = parsed_batch_dir / f"{batch_id}.json"

        # Skip if all record outputs exist
        if not args.force:
            all_exist = True
            for p in paths:
                rid = str(load_json(p).get("record_id"))
                if not (parsed_record_dir / f"rid={rid}.json").exists():
                    all_exist = False
                    break
            if all_exist:
                print(f"[SKIP] {batch_id} (all record outputs exist)")
                ok_batches += 1
                continue

        inp_records = [load_json(p) for p in paths]
        user_payload = {"records": inp_records}

        try:
            resp = call_with_retry(
                client=client,
                model=DEFAULT_MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_payload=user_payload,
                schema=OUTPUT_SCHEMA,
                prompt_cache_key=args.cache_key,
                max_retries=args.max_retries,
                base_sleep=args.base_sleep,
            )

            # Save raw response
            try:
                raw = resp.model_dump()
            except Exception:
                raw = {"_repr": repr(resp)}
            dump_json(out_resp_path, raw)

            # Parse JSON text (already schema-validated by Structured Outputs)
            out_text = extract_output_text(resp)
            out_obj = json.loads(out_text)
            out_obj = sanitize(out_obj, inp_records)
            dump_json(out_parsed_batch_path, out_obj)

            # Split per record for downstream
            parsed_record_dir.mkdir(parents=True, exist_ok=True)
            for x in out_obj.get("results", []):
                rid = str(x.get("record_id"))
                dump_json(parsed_record_dir / f"rid={rid}.json", x)

            print(f"[OK] {batch_id} -> {out_parsed_batch_path.name} (records={len(paths)})")
            ok_batches += 1

        except Exception as e:
            ng_batches += 1
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / f"{batch_id}.txt").write_text(str(e), encoding="utf-8")
            print(f"[NG] {batch_id}: {type(e).__name__}: {e}")

    print(f"[DONE] ok_batches={ok_batches} ng_batches={ng_batches} out_dir={out_dir}")


if __name__ == "__main__":
    main()
