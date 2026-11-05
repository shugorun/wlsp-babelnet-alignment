"""Asking an LLM which candidate synsets of a WLSP record are equivalent to it.

A prompt is a directory of prompts/alignment/ holding system.txt (the
instructions), schema.json (the JSON schema of the answer) and config.json:

    mode        independent: a label for every candidate;
                set: the list of the EQUAL candidates, possibly empty;
                rank_verify: set, then the candidates it chose are given
                to the prompt named by verify_version, and stay EQUAL
                only if that prompt labels them EQUAL as well
    chunk       the most candidates given in one request
    wlsp_field  the record text given: wlsp_min or wlsp_full
    bn_field    the synset text given: bn_1step or bn_full
    cand_order  the order of the candidates: as in the data (the default),
                reverse, or shuffle (seeded by the record ID)

Each request holds one record and up to `chunk` of its candidates, as the
JSON user message

    {"record_id": "...", "chunk": {"index": i},
     "wlsp": {"text": "..."},
     "candidates": [{"synset_id": "bn:...", "text": "..."}, ...]}

An answer is accepted after a few repairs: a candidate labelled twice keeps
its first label; one wrong synset ID is put right when exactly one candidate
is missing; and missing candidates are labelled NONE, up to one in twenty of
the candidates of the request and always at least one. Any other mismatch,
or a label outside ALLOWED_LABELS, rejects the answer.
"""

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ALLOWED_LABELS = {"EQUAL", "HYPERNYM", "HYPONYM", "NONE"}
SCHEMA_NAME = "wlsp_babelnet_alignment"
OPENAI_ATTEMPTS = 5  # tries of a request that meets a rate limit or server error

# A call takes the system prompt, the schema and the request, and returns
# the parsed answer.
Call = Callable[[str, dict, dict], dict]


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    schema: dict
    config: dict
    verify: "Prompt | None" = None

    @property
    def mode(self) -> str:
        return self.config["mode"]


def load_prompt(directory: Path) -> Prompt:
    """Read a prompt, and its second-pass prompt if it has one.

    A second-pass prompt has only system.txt and schema.json: the requests
    it gets are made by the first pass.
    """
    config_file = directory / "config.json"
    config = (
        json.loads(config_file.read_text(encoding="utf-8"))
        if config_file.exists()
        else {}
    )
    verify = None
    if config.get("mode") == "rank_verify":
        verify = load_prompt(directory.parent / config["verify_version"])
    return Prompt(
        name=directory.name,
        system=(directory / "system.txt").read_text(encoding="utf-8"),
        schema=json.loads((directory / "schema.json").read_text(encoding="utf-8")),
        config=config,
        verify=verify,
    )


def requests(records: list[dict], prompt: Prompt) -> list[dict]:
    """Split each record's candidates into the requests the prompt makes.

    A record is {"record_id": str, "wlsp": {field: text},
    "candidates": [{"synset_id": str, field: text, ...}]}.
    """
    size = prompt.config["chunk"]
    wlsp_field = prompt.config["wlsp_field"]
    bn_field = prompt.config["bn_field"]
    order = prompt.config.get("cand_order", "input")
    out = []
    for record in records:
        candidates = list(record["candidates"])
        if order == "reverse":
            candidates.reverse()
        elif order == "shuffle":
            random.Random(record["record_id"]).shuffle(candidates)
        elif order != "input":
            raise ValueError(f"unknown cand_order: {order}")
        for i in range(0, len(candidates), size):
            out.append(
                {
                    "record_id": record["record_id"],
                    "chunk": {"index": i // size},
                    "wlsp": {"text": record["wlsp"][wlsp_field]},
                    "candidates": [
                        {"synset_id": c["synset_id"], "text": c[bn_field]}
                        for c in candidates[i : i + size]
                    ],
                }
            )
    return out


def _normalized_id(sid: str) -> str:
    """The synset ID without leading zeros, to match an ID that lost or gained some."""
    body = sid[3:] if sid.startswith("bn:") else sid
    if len(body) < 2:
        return sid
    return "bn:" + body[:-1].lstrip("0") + body[-1]


def labels_from_list(answer: dict, request: dict, fill: bool = True) -> dict[str, str]:
    """The label of each candidate from an answer that labels every one."""
    labels = answer.get("labels")
    if not isinstance(labels, list):
        raise ValueError("labels missing")
    seen, kept = set(), []
    for item in labels:
        if item.get("synset_id") not in seen:
            seen.add(item.get("synset_id"))
            kept.append(item)
    labels = kept

    asked = {c["synset_id"] for c in request["candidates"]}
    given = {item.get("synset_id") for item in labels}
    missing, extra = asked - given, given - asked
    if len(missing) == 1 and len(extra) == 1:
        wrong, right = next(iter(extra)), next(iter(missing))
        for item in labels:
            if item.get("synset_id") == wrong:
                item["synset_id"] = right
                break
    given = {item.get("synset_id") for item in labels}
    still = [sid for sid in asked if sid not in given]
    if fill and still and len(still) <= max(1, len(request["candidates"]) // 20):
        for sid in still:
            labels.append({"synset_id": sid, "label": "NONE"})
    given = {item.get("synset_id") for item in labels}
    if given != asked:
        raise ValueError(
            f"synset set mismatch: missing={asked - given} extra={given - asked}"
        )
    for item in labels:
        if item.get("label") not in ALLOWED_LABELS:
            raise ValueError(f"invalid label {item.get('label')}")
    return {item["synset_id"]: item["label"] for item in labels}


def labels_from_set(answer: dict, request: dict) -> dict[str, str]:
    """The label of each candidate from an answer that lists the EQUAL ones.

    A listed ID that matches no candidate even without leading zeros is
    dropped, up to one in ten of the candidates and always at least one.
    """
    ids = answer.get("equal_synset_ids")
    if not isinstance(ids, list):
        raise ValueError("equal_synset_ids missing")
    asked = {c["synset_id"] for c in request["candidates"]}
    by_normalized = {_normalized_id(s): s for s in asked}
    chosen, unknown = set(), 0
    for sid in ids:
        if sid in asked:
            chosen.add(sid)
        elif _normalized_id(str(sid)) in by_normalized:
            chosen.add(by_normalized[_normalized_id(str(sid))])
        else:
            unknown += 1
    if unknown > max(1, len(asked) // 10):
        raise ValueError(f"too many unknown ids: {unknown}")
    return {sid: ("EQUAL" if sid in chosen else "NONE") for sid in asked}


def judge(call: Call, prompt: Prompt, request: dict) -> dict[str, str]:
    """The label of each candidate of one request."""
    answer = call(prompt.system, prompt.schema, request)
    if prompt.mode == "independent":
        labels = labels_from_list(answer, request)
    else:
        labels = labels_from_set(answer, request)
    if prompt.mode == "rank_verify":
        chosen = [s for s, label in labels.items() if label == "EQUAL"]
        if chosen:
            second = {
                "record_id": request["record_id"],
                "wlsp": request["wlsp"],
                "candidates": [
                    c for c in request["candidates"] if c["synset_id"] in chosen
                ],
            }
            answer = call(prompt.verify.system, prompt.verify.schema, second)
            # The second pass sees few candidates, so none is filled in.
            confirmed = labels_from_list(answer, second, fill=False)
            for sid in chosen:
                if confirmed.get(sid) != "EQUAL":
                    labels[sid] = "NONE"
    return labels


def user_message(request: dict) -> str:
    return json.dumps(request, ensure_ascii=False)


def vllm_call(client, model: str, thinking: bool, temperature: float) -> Call:
    """A call to a vLLM server through its OpenAI-compatible chat API.

    With a temperature above 0, the sampling recommended for Qwen3.5 is
    used: top_p 0.95 and top_k 20.
    """
    extra: dict = {"chat_template_kwargs": {"enable_thinking": thinking}}
    sampling: dict = {"temperature": temperature}
    if temperature > 0:
        sampling["top_p"] = 0.95
        extra["top_k"] = 20

    def call(system: str, schema: dict, request: dict):
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message(request)},
            ],
            max_tokens=None,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "schema": schema},
            },
            extra_body=extra,
            **sampling,
        )
        return json.loads(r.choices[0].message.content)

    return call


def openai_call(client, model: str, effort: str) -> Call:
    """A call to the OpenAI Responses API with the given reasoning effort.

    The temperature is left at the API's default, the only one reasoning
    models accept. Rate limits and server errors are retried with
    exponential backoff; other client errors are raised at once.
    """

    def call(system: str, schema: dict, request: dict):
        for attempt in range(1, OPENAI_ATTEMPTS + 1):
            try:
                r = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message(request)},
                    ],
                    reasoning={"effort": effort},
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": SCHEMA_NAME,
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    prompt_cache_key=SCHEMA_NAME,
                )
                return json.loads(r.output_text)
            except Exception as e:
                status = getattr(e, "status_code", None)
                client_error = isinstance(status, int) and 400 <= status < 500
                if (client_error and status != 429) or attempt == OPENAI_ATTEMPTS:
                    raise
                time.sleep(2 ** (attempt - 1) + random.random() * 0.2)

    return call
