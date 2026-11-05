"""Label the candidate synsets of each record with an LLM.

Reads data/interim/pairs_{a,b}.parquet and, for Dataset B, the split in
data/splits/dataset_b.tsv, and writes the labels of each run to

    data/predictions/<data>/<model>/<condition>/run<k>.tsv

with the columns record_id, synset_id and prediction (EQUAL, NONE, or for
the prompts that use them, HYPERNYM or HYPONYM). <data> is "a", "b-dev" or
"b-test", followed by the query types when only their candidates are
judged; <condition> is the prompt, followed by "-thinking" and "-t<T>" when
Qwen thinks or samples at temperature T, or by the reasoning effort of GPT
when it is not none.

Qwen3.5-27B is served by vLLM (docs/evaluation.md says how), and GPT-5.4
is called through the OpenAI API with OPENAI_API_KEY set:

    uv run python scripts/evaluation/02_judge.py --data b --split test \\
        --prompt v0 --model qwen3.5-27b
    uv run python scripts/evaluation/02_judge.py --data b --split test \\
        --prompt v0 --api openai --model gpt-5.4-2026-03-05

Each answer is appended to run<k>.progress.jsonl as it comes. A request
that fails twice is left out of the run, which then stops; running the same
command again sends only the requests that are missing.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

from wlsp_bn import paths
from wlsp_bn.alignment import (
    Call,
    Prompt,
    judge,
    load_prompt,
    openai_call,
    requests,
    vllm_call,
)

TIMEOUT = 1800  # seconds a request may take; a thinking model needs long


def load_records(data: str, split: str | None, query_types: list[str]) -> list[dict]:
    """The records to judge, each with its candidates, in the order of the pairs."""
    pairs = pd.read_parquet(paths.INTERIM / f"pairs_{data}.parquet")
    if split:
        splits = pd.read_csv(paths.SPLITS / "dataset_b.tsv", sep="\t")
        chosen = splits.loc[splits["split"] == split, "record_id"]
        pairs = pairs[pairs["record_id"].isin(chosen)]
    if query_types:
        pairs = pairs[pairs["query_type"].isin(query_types)]
    records: dict[int, dict] = {}
    for p in pairs.itertuples():
        record = records.setdefault(
            p.record_id,
            {
                "record_id": str(p.record_id),
                "wlsp": {"wlsp_min": p.wlsp_min, "wlsp_full": p.wlsp_full},
                "candidates": [],
            },
        )
        record["candidates"].append(
            {"synset_id": p.synset_id, "bn_1step": p.bn_1step, "bn_full": p.bn_full}
        )
    return list(records.values())


def judge_all(
    call: Call, prompt: Prompt, chunks: list[dict], progress: Path, workers: int
) -> list[dict]:
    """Judge the requests not yet in `progress`; return the answer to every request."""
    done = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["error"]:
                done[row["record_id"], row["chunk_index"]] = row
    todo = [c for c in chunks if (c["record_id"], c["chunk"]["index"]) not in done]
    print(f"{progress.parent}: {len(chunks)} requests, {len(todo)} to send", flush=True)

    def process(request: dict) -> dict:
        key = {
            "record_id": request["record_id"],
            "chunk_index": request["chunk"]["index"],
        }
        for _ in range(2):
            try:
                labels = judge(call, prompt, request)
                break
            except Exception as e:
                error = e
        else:
            print(
                f"  failed {key}: {type(error).__name__}: {str(error)[:120]}",
                flush=True,
            )
            return {**key, "labels": None, "error": type(error).__name__}
        return {
            **key,
            "labels": [{"synset_id": s, "label": v} for s, v in labels.items()],
            "error": None,
        }

    lock = threading.Lock()
    sent = [0]

    def save(row: dict, f) -> None:
        with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if not row["error"]:
                done[row["record_id"], row["chunk_index"]] = row
            sent[0] += 1
            if sent[0] % 20 == 0:
                print(f"  {sent[0]}/{len(todo)}", flush=True)

    pool = ThreadPoolExecutor(max_workers=workers)
    with progress.open("a", encoding="utf-8") as f, pool:
        for row in pool.map(process, todo):
            save(row, f)

    missing = [c for c in chunks if (c["record_id"], c["chunk"]["index"]) not in done]
    if missing:
        raise SystemExit(
            f"{len(missing)} requests failed; run the same command again to resend them"
        )
    return [done[c["record_id"], c["chunk"]["index"]] for c in chunks]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", choices=["a", "b"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], help="required for b")
    ap.add_argument("--query-types", nargs="+", default=[], metavar="TYPE")
    ap.add_argument("--prompt", required=True, help="a directory of prompts/alignment/")
    ap.add_argument("--api", choices=["vllm", "openai"], default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vllm")
    ap.add_argument("--thinking", action="store_true", help="vllm")
    ap.add_argument("--temperature", type=float, default=0.0, help="vllm")
    ap.add_argument("--effort", default="none", help="openai: the reasoning effort")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if (args.data == "b") != bool(args.split):
        ap.error("--split is required for b and not used for a")

    prompt = load_prompt(paths.PROMPTS / "alignment" / args.prompt)
    records = load_records(args.data, args.split, args.query_types)
    chunks = requests(records, prompt)
    if args.api == "vllm":
        client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=TIMEOUT)
        call = vllm_call(client, args.model, args.thinking, args.temperature)
        condition = args.prompt + "-thinking" * args.thinking
        condition += f"-t{args.temperature:g}" if args.temperature > 0 else ""
    else:
        call = openai_call(OpenAI(timeout=TIMEOUT), args.model, args.effort)
        condition = args.prompt + (f"-{args.effort}" if args.effort != "none" else "")

    data = "-".join([args.data, *filter(None, [args.split]), *args.query_types])
    out_dir = paths.PREDICTIONS / data / args.model / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(1, args.runs + 1):
        out = out_dir / f"run{k}.tsv"
        if out.exists():
            print(f"{out} exists")
            continue
        answers = judge_all(
            call, prompt, chunks, out.with_suffix(".progress.jsonl"), args.workers
        )
        labels = {
            (a["record_id"], x["synset_id"]): x["label"]
            for a in answers
            for x in a["labels"]
        }
        rows = [
            (
                int(r["record_id"]),
                c["synset_id"],
                labels[r["record_id"], c["synset_id"]],
            )
            for r in records
            for c in r["candidates"]
        ]
        predictions = pd.DataFrame(
            rows, columns=["record_id", "synset_id", "prediction"]
        )
        predictions.to_csv(out, sep="\t", index=False, lineterminator="\n")
        n_eq = int((predictions["prediction"] == "EQUAL").sum())
        print(f"{out}: {len(predictions):,} pairs, {n_eq} EQUAL")


if __name__ == "__main__":
    main()
