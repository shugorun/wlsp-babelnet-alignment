"""List the query terms that retrieve the candidate synsets of each record.

Reads data/interim/wlsp.parquet and data/interim/samples.json and writes
data/interim/queries.json: for each dataset and record, the terms of each
query type. Both datasets query BabelNet with

    headword  the WLSP headword (Japanese)
    jmdict    the English terms that JMdict gives for the headword

and Dataset B also with the Japanese and English aliases and spelling
variants that GPT-5.4 generated for the record:

    alias_ja, alias_en, variant_ja, variant_en

The generated terms are kept in data/term_expansion/dataset_b.jsonl, which
is part of the repository, so that the candidates can be retrieved again
without calling the model. With --call-api the terms of records missing from
that file are generated (OPENAI_API_KEY must be set). The prompt and output
schema are in prompts/term_expansion/; the prompt also asks for hypernyms,
which were not used as queries and are not kept.

    uv run python scripts/dataset/03_make_queries.py [--call-api]
"""

import argparse
import json
import re

import pandas as pd

from wlsp_bn import paths
from wlsp_bn.cells import unique

WLSP = paths.INTERIM / "wlsp.parquet"
SAMPLES = paths.INTERIM / "samples.json"
EXPANSION = paths.TERM_EXPANSION / "dataset_b.jsonl"
OUT = paths.INTERIM / "queries.json"
PROMPT = paths.PROMPTS / "term_expansion"

MODEL = "gpt-5.4-2026-03-05"
BATCH_SIZE = 20
# At most this many JMdict terms and co-classified words are given to the model.
MAX_CONTEXT_TERMS = 12
# The generated fields kept, and the query type each one becomes.
GENERATED = {
    "aliases_ja": "alias_ja",
    "aliases_en": "alias_en",
    "variants_ja": "variant_ja",
    "variants_en": "variant_en",
}
MAX_GENERATED = 3


def model_input(record_id: int, record: pd.Series) -> dict:
    """The description of one record that the model expands."""
    headword = record["headword"].strip()
    path = [
        str(record[level]).strip()
        for level in ["division", "section", "category"]
        if pd.notna(record[level]) and str(record[level]).strip()
    ]
    reading = record["reading"].strip() if isinstance(record["reading"], str) else ""
    co_classified = [t for t in unique(list(record["co_classified"])) if t != headword]
    return {
        "record_id": record_id,
        "category_path_ja": ">".join(path),
        "lemma_ja": headword,
        "kana": reading or None,
        "jmdict_en": unique(list(record["jmdict_en"]))[:MAX_CONTEXT_TERMS],
        "subparagraph_terms_ja": co_classified[:MAX_CONTEXT_TERMS],
    }


def drop_repeated_inputs(result: dict, given: dict) -> dict:
    """Keep up to three generated terms per field, none repeating an input term."""
    english = {re.sub(r"\s+", " ", t.strip().lower()) for t in given["jmdict_en"]}
    kept = {"record_id": int(result["record_id"])}
    for field in GENERATED:
        terms = [t for t in result[field] if isinstance(t, str)]
        if field.endswith("_ja"):
            terms = [t for t in terms if t != given["lemma_ja"]]
        else:
            terms = [
                t
                for t in terms
                if re.sub(r"\s+", " ", t.strip().lower()) not in english
            ]
        kept[field] = terms[:MAX_GENERATED]
    return kept


def expand(inputs: list[dict]) -> list[dict]:
    """Generate aliases and variants with GPT-5.4, BATCH_SIZE records per call."""
    from openai import OpenAI

    client = OpenAI(max_retries=5)
    system = (PROMPT / "system.txt").read_text(encoding="utf-8")
    schema = json.loads((PROMPT / "schema.json").read_text(encoding="utf-8"))
    results = []
    for start in range(0, len(inputs), BATCH_SIZE):
        batch = inputs[start : start + BATCH_SIZE]
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps({"records": batch}, ensure_ascii=False),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "wlsp_fallback_terms_batch_v4",
                    "schema": schema,
                    "strict": True,
                }
            },
            reasoning={"effort": "high"},
        )
        given = {str(r["record_id"]): r for r in batch}
        for result in json.loads(response.output_text)["results"]:
            results.append(
                drop_repeated_inputs(result, given[str(result["record_id"])])
            )
        print(f"expanded {start + len(batch)}/{len(inputs)} records")
    return results


def generated_terms(
    records: pd.DataFrame, ids: list[int], call_api: bool
) -> dict[int, dict]:
    """The generated terms of Dataset B, from EXPANSION or, if asked, the model."""
    done = {}
    if EXPANSION.exists():
        for line in EXPANSION.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["record_id"]] = row
    # The records were sent in the order of their IDs as strings.
    missing = sorted((i for i in ids if i not in done), key=str)
    if missing and not call_api:
        raise SystemExit(
            f"{len(missing)} records have no generated terms in {EXPANSION}; "
            "run with --call-api to generate them"
        )
    if missing:
        new = expand([model_input(i, records.loc[i]) for i in missing])
        EXPANSION.parent.mkdir(parents=True, exist_ok=True)
        with EXPANSION.open("a", encoding="utf-8") as f:
            for row in new:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[row["record_id"]] = row
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--call-api", action="store_true", help="generate missing terms with GPT-5.4"
    )
    args = ap.parse_args()

    records = pd.read_parquet(WLSP)
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    generated = generated_terms(records, samples["b"], args.call_api)

    queries: dict[str, dict[int, dict[str, list[str]]]] = {"a": {}, "b": {}}
    for dataset, ids in samples.items():
        for i in ids:
            record = records.loc[i]
            terms = {
                "headword": [record["headword"]],
                "jmdict": list(record["jmdict_en"]),
            }
            if dataset == "b":
                terms |= {qt: generated[i][field] for field, qt in GENERATED.items()}
            queries[dataset][i] = terms
    OUT.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT}: queries of {len(queries['a'])} + {len(queries['b'])} records")


if __name__ == "__main__":
    main()
