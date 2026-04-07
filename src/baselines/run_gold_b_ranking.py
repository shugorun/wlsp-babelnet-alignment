"""Rank gold_B candidates with multilingual E5 embeddings.

This script rebuilds the ranking baseline with multiple query/passage text
templates. It keeps the original concise/natural templates and also adds
templates inspired by the ranking implementation in ../final.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.experiment_log import append_run_log

ROOT = Path(__file__).resolve().parents[2]

RECORDS_PATH = ROOT / "data" / "gold" / "gold_B_records.pkl"
GOLD_PATH = ROOT / "data" / "gold" / "gold_B.pkl"
BABELNET_PATH = ROOT / "data" / "processed" / "babelnet_.pkl"

OUTPUT_DIR = ROOT / "outputs" / "baselines" / "gold_B_ranking"
RANKINGS_PATH = OUTPUT_DIR / "rankings.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.pkl"

MODEL_NAME = "intfloat/multilingual-e5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_LENGTH = 384
TOPKS = [1, 2, 3, 5]

QUERY_MODES = [
    "headword",
    "headword_class",
    "headword_synonyms",
    "headword_class_synonyms",
    "sentence_headword_class_synonyms",
    "query_sentence_headword_class_synonyms",
]

PASSAGE_MODES = [
    "gloss",
    "gloss_lemmas",
    "gloss_lemmas_categories",
    "sentence_gloss_lemmas",
    "passage_sentence_gloss_lemmas",
]

SYMMETRIC_MODES = ["concise", "natural"]

# The ranking script compares several text templates, including compact
# symmetric prompts and asymmetric query/passage templates.

def normalize_ja_text(text: object) -> str:
    """Normalize a Japanese-like text cell."""
    if text is None or pd.isna(text):
        return ""
    out = str(text).strip()
    out = re.sub(r"\s+", " ", out)
    return out


def normalize_en_text(text: object) -> str:
    """Normalize an English-like text cell."""
    if text is None or pd.isna(text):
        return ""
    out = str(text).strip().lower()
    out = out.replace("_", " ")
    out = re.sub(r"\s+", " ", out)
    return out


def normalize_list(value: object, normalizer) -> list[str]:
    """Convert one cell to a clean string list."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif not isinstance(value, (list, tuple, set, pd.Series, pd.Index)):
        value = [value]

    out = []
    for item in value:
        if item is None or pd.isna(item):
            continue
        text = normalizer(item)
        if text:
            out.append(text)
    return out


def dedupe_keep_order(items: list[str]) -> list[str]:
    """Return unique strings while keeping order."""
    return list(dict.fromkeys(items))


def build_query_text(row: pd.Series, mode: str) -> str:
    """Build one WLSP-side query text."""
    division = normalize_ja_text(row.get("division", ""))
    category = normalize_ja_text(row.get("category", ""))
    subcategory = normalize_ja_text(row.get("subcategory", ""))
    lemma = normalize_ja_text(row.get("lemma", ""))
    class_name = normalize_ja_text(row.get("class", ""))
    kana = normalize_ja_text(row.get("kana", ""))
    synonyms = normalize_list(row.get("synonyms", []), normalize_ja_text)
    en_terms = normalize_list(row.get("EN_JMdict", []), normalize_en_text)

    if mode == "concise":
        parts = [f"query: {lemma}"]
        if class_name:
            parts.append(f"class: {class_name}")
        if subcategory and subcategory != class_name:
            parts.append(f"subcategory: {subcategory}")
        if category and category not in {subcategory, class_name}:
            parts.append(f"category: {category}")
        if division and division not in {category, subcategory, class_name}:
            parts.append(f"division: {division}")
        if kana:
            parts.append(f"reading: {kana}")
        if synonyms:
            parts.append("related terms: " + " | ".join(synonyms[:5]))
        if en_terms:
            parts.append("english: " + " | ".join(en_terms[:2]))
        return "; ".join(parts)

    if mode == "natural":
        parts = [f"query: {lemma}"]
        category_parts = [part for part in [division, category, subcategory, class_name] if part]
        if category_parts:
            parts.append("belongs to: " + " > ".join(dict.fromkeys(category_parts)))
        if kana:
            parts.append(f"reading: {kana}")
        if synonyms:
            parts.append("related terms: " + " | ".join(synonyms[:5]))
        if en_terms:
            parts.append("english headwords: " + " | ".join(en_terms[:2]))
        return ". ".join(parts)

    if mode == "headword":
        return f"query: headword={lemma}"

    if mode == "headword_class":
        return f"query: headword={lemma}; class={class_name}" if class_name else f"query: headword={lemma}"

    if mode == "headword_synonyms":
        if synonyms:
            return f"query: headword={lemma}; synonyms={'|'.join(synonyms[:5])}"
        return f"query: headword={lemma}"

    if mode == "headword_class_synonyms":
        parts = [f"query: headword={lemma}"]
        if class_name:
            parts.append(f"class={class_name}")
        if synonyms:
            parts.append(f"synonyms={'|'.join(synonyms[:5])}")
        return "; ".join(parts)

    if mode == "sentence_headword_class_synonyms":
        parts = [f"{lemma} is a WLSP headword"]
        if class_name:
            parts.append(f"in class {class_name}")
        if synonyms:
            parts.append("and related to " + " | ".join(synonyms[:5]))
        return " ".join(parts)

    if mode == "query_sentence_headword_class_synonyms":
        parts = [f"query: {lemma} is a WLSP headword"]
        if class_name:
            parts.append(f"in class {class_name}")
        if synonyms:
            parts.append("and related to " + " | ".join(synonyms[:5]))
        return " ".join(parts)

    raise ValueError(f"Unknown query mode: {mode}")


def build_passage_text(row: pd.Series, mode: str) -> str:
    """Build one BabelNet-side passage text."""
    lemmas_ja = normalize_list(row.get("lemmas_JA", []), normalize_ja_text)
    lemmas_en = normalize_list(row.get("lemmas_EN", []), normalize_en_text)
    categories_ja = normalize_list(row.get("categories_JA", []), normalize_ja_text)
    categories_en = normalize_list(row.get("categories_EN", []), normalize_en_text)
    main_gloss_ja = normalize_ja_text(row.get("main_gloss_JA", ""))
    main_gloss_en = normalize_en_text(row.get("main_gloss_EN", ""))

    lemma = lemmas_ja[0] if lemmas_ja else (lemmas_en[0] if lemmas_en else "")
    synonyms = lemmas_ja[1:4] if lemmas_ja else lemmas_en[1:4]
    gloss = main_gloss_ja if main_gloss_ja else main_gloss_en
    categories = categories_ja[:3] if categories_ja else categories_en[:3]

    if mode == "concise":
        parts = [f"passage: {gloss}" if gloss else f"passage: {lemma}"]
        if lemma:
            parts.append(f"headword: {lemma}")
        if gloss:
            parts.append(f"definition: {gloss}")
        if synonyms:
            parts.append("related lemmas: " + " | ".join(synonyms))
        if categories:
            parts.append("categories: " + " | ".join(categories))
        return "; ".join(parts)

    if mode == "natural":
        parts = [f"passage: {lemma}" if lemma else "passage:"]
        if gloss:
            parts.append(f"definition: {gloss}")
        if synonyms:
            parts.append("related lemmas: " + " | ".join(synonyms))
        if categories:
            parts.append("categories: " + " | ".join(categories))
        return ". ".join(parts)

    if mode == "gloss":
        return f'passage: gloss="{gloss}"' if gloss else f"passage: {lemma}"

    if mode == "gloss_lemmas":
        parts = [f'passage: gloss="{gloss}"' if gloss else f"passage: {lemma}"]
        if lemma:
            lemma_part = "|".join([lemma] + synonyms)
            parts.append(f"lemmas={lemma_part}")
        return "; ".join(parts)

    if mode == "gloss_lemmas_categories":
        parts = [f'passage: gloss="{gloss}"' if gloss else f"passage: {lemma}"]
        if lemma:
            lemma_part = "|".join([lemma] + synonyms)
            parts.append(f"lemmas={lemma_part}")
        if categories:
            parts.append(f"categories={'|'.join(categories)}")
        return "; ".join(parts)

    if mode == "sentence_gloss_lemmas":
        parts = [f"{lemma} is a BabelNet lemma" if lemma else "This is a BabelNet synset"]
        if gloss:
            parts.append(f'whose gloss is "{gloss}"')
        if synonyms:
            parts.append("and related lemmas are " + " | ".join(synonyms))
        return " ".join(parts)

    if mode == "passage_sentence_gloss_lemmas":
        parts = [f"passage: {lemma} is a BabelNet lemma" if lemma else "passage:"]
        if gloss:
            parts.append(f'whose gloss is "{gloss}"')
        if synonyms:
            parts.append("and related lemmas are " + " | ".join(synonyms))
        return " ".join(parts)

    raise ValueError(f"Unknown passage mode: {mode}")


def get_mode_specs() -> list[tuple[str, str, str]]:
    """Return (text_mode, query_mode, passage_mode) specs."""
    specs = [(mode, mode, mode) for mode in SYMMETRIC_MODES]
    for query_mode in QUERY_MODES:
        for passage_mode in PASSAGE_MODES:
            specs.append((f"{query_mode}__{passage_mode}", query_mode, passage_mode))
    return specs


@torch.inference_mode()
def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Encode texts with multilingual E5."""
    vecs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(DEVICE)

        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        pooled = F.normalize(pooled, p=2, dim=1)
        vecs.extend(pooled.detach().cpu().numpy().astype(np.float32))

    return np.stack(vecs, axis=0)


def encode_by_mode(
    model,
    tokenizer,
    records: pd.DataFrame,
    babelnet: pd.DataFrame,
    record_ids: list[int],
    candidate_sids: list[str],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Pre-encode query and passage texts for all modes."""
    query_vecs_by_mode = {}
    passage_vecs_by_mode = {}

    query_modes = sorted(set(SYMMETRIC_MODES + QUERY_MODES))
    passage_modes = sorted(set(SYMMETRIC_MODES + PASSAGE_MODES))

    # Pre-encoding all templates once keeps the later ranking loop simple.
    for mode in query_modes:
        print(f"Building query vectors for query_mode={mode}")
        texts = [build_query_text(records.loc[rid], mode) for rid in record_ids]
        vecs = encode_texts(model, tokenizer, texts)
        query_vecs_by_mode[mode] = {rid: vecs[i] for i, rid in enumerate(record_ids)}

    for mode in passage_modes:
        print(f"Building passage vectors for passage_mode={mode}")
        texts = [build_passage_text(babelnet.loc[sid], mode) for sid in candidate_sids]
        vecs = encode_texts(model, tokenizer, texts)
        passage_vecs_by_mode[mode] = {sid: vecs[i] for i, sid in enumerate(candidate_sids)}

    return query_vecs_by_mode, passage_vecs_by_mode


def build_rankings(
    records: pd.DataFrame,
    gold: pd.DataFrame,
    babelnet: pd.DataFrame,
) -> pd.DataFrame:
    """Build candidate rankings for each template mode and candidate mode."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE).eval()

    record_ids = records.index.astype(int).tolist()
    candidate_sids = sorted({str(sid) for sid in gold["synset_id"].astype(str).tolist()})
    mode_specs = get_mode_specs()

    query_vecs_by_mode, passage_vecs_by_mode = encode_by_mode(
        model,
        tokenizer,
        records,
        babelnet,
        record_ids,
        candidate_sids,
    )

    rows = []
    for text_mode, query_mode, passage_mode in mode_specs:
        print(f"Scoring ranking for text_mode={text_mode}")

        for rid, row in tqdm(records.iterrows(), total=len(records), desc=f"Ranking {text_mode}", unit="record"):
            rid = int(rid)
            query_vec = query_vecs_by_mode[query_mode][rid]

            sids_ja = dedupe_keep_order([str(sid) for sid in normalize_list(row.get("sids_JA", []), str)])
            sids_jm = dedupe_keep_order([str(sid) for sid in normalize_list(row.get("sids_JMdict", []), str)])

            candidate_specs = {
                "union": dedupe_keep_order(sids_ja + sids_jm),
                "ja_preferred": sids_ja if sids_ja else sids_jm,
            }

            # Compare two candidate policies:
            # - union: JA and JMdict candidates together
            # - ja_preferred: JA only when available, otherwise JMdict
            for candidate_mode, candidate_sids_for_row in candidate_specs.items():
                if not candidate_sids_for_row:
                    continue

                scores = []
                for sid in candidate_sids_for_row:
                    if sid not in passage_vecs_by_mode[passage_mode]:
                        continue
                    score = float(np.dot(query_vec, passage_vecs_by_mode[passage_mode][sid]))
                    scores.append((sid, score))

                if not scores:
                    continue

                scores.sort(key=lambda item: item[1], reverse=True)
                for rank, (sid, score) in enumerate(scores, start=1):
                    rows.append(
                        {
                            "record_id": rid,
                            "text_mode": text_mode,
                            "query_mode": query_mode,
                            "passage_mode": passage_mode,
                            "candidate_mode": candidate_mode,
                            "rank": rank,
                            "synset_id": sid,
                            "score": score,
                        }
                    )

    return pd.DataFrame(rows)


def evaluate_topk(rankings: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Evaluate top-k EQUAL F1 for each text mode and candidate mode."""
    gold_eval = gold.copy()
    gold_eval["record_id"] = gold_eval["record_id"].astype(int)
    gold_eval["synset_id"] = gold_eval["synset_id"].astype(str)
    gold_eval["label_bin"] = (gold_eval["label"].astype(str) == "EQUAL").astype(int)

    rows = []
    for text_mode in sorted(rankings["text_mode"].unique()):
        rankings_text = rankings[rankings["text_mode"] == text_mode].copy()

        for candidate_mode in sorted(rankings_text["candidate_mode"].unique()):
            ranking_mode = rankings_text[rankings_text["candidate_mode"] == candidate_mode].copy()

            for topk in TOPKS:
                pred_pairs = ranking_mode[ranking_mode["rank"] <= topk][["record_id", "synset_id"]].copy()
                pred_pairs["pred_bin"] = 1

                merged = gold_eval.merge(pred_pairs, on=["record_id", "synset_id"], how="left")
                merged["pred_bin"] = merged["pred_bin"].fillna(0).astype(int)

                tp = int(((merged["label_bin"] == 1) & (merged["pred_bin"] == 1)).sum())
                fp = int(((merged["label_bin"] == 0) & (merged["pred_bin"] == 1)).sum())
                fn = int(((merged["label_bin"] == 1) & (merged["pred_bin"] == 0)).sum())

                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

                rows.append(
                    {
                        "text_mode": text_mode,
                        "query_mode": ranking_mode["query_mode"].iloc[0],
                        "passage_mode": ranking_mode["passage_mode"].iloc[0],
                        "candidate_mode": candidate_mode,
                        "topk": topk,
                        "precision_equal": precision,
                        "recall_equal": recall,
                        "f1_equal": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["f1_equal", "precision_equal", "recall_equal"],
        ascending=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the same three tables as the lexical baseline, then rank the
    # gold-defined candidate set with E5 similarities.
    records = pd.read_pickle(RECORDS_PATH)
    gold = pd.read_pickle(GOLD_PATH)
    babelnet = pd.read_pickle(BABELNET_PATH)
    babelnet.index = babelnet.index.map(str)

    rankings = build_rankings(records, gold, babelnet)
    rankings.to_pickle(RANKINGS_PATH)
    print(f"Saved rankings: {RANKINGS_PATH}")

    metrics = evaluate_topk(rankings, gold)
    with METRICS_PATH.open("wb") as f:
        pickle.dump({"summary": metrics}, f)
    print(f"Saved metrics: {METRICS_PATH}")

    print(metrics.head(20).to_string(index=False))

    best_row = metrics.iloc[0].to_dict()
    append_run_log(
        run_name="gold_B_ranking_e5",
        rationale="Rebuild the E5 ranking baseline on gold_B and compare multiple query / passage templates, including templates inspired by ../final.",
        script_path="src/baselines/run_gold_b_ranking.py",
        params={
            "model_name": MODEL_NAME,
            "n_query_modes": len(sorted(set(SYMMETRIC_MODES + QUERY_MODES))),
            "n_passage_modes": len(sorted(set(SYMMETRIC_MODES + PASSAGE_MODES))),
            "n_text_modes": len(get_mode_specs()),
            "topks": ",".join(str(k) for k in TOPKS),
            "max_length": MAX_LENGTH,
        },
        metrics={
            "best_text_mode": best_row["text_mode"],
            "best_query_mode": best_row["query_mode"],
            "best_passage_mode": best_row["passage_mode"],
            "best_candidate_mode": best_row["candidate_mode"],
            "best_topk": best_row["topk"],
            "best_pair_f1": best_row["f1_equal"],
            "best_pair_precision": best_row["precision_equal"],
            "best_pair_recall": best_row["recall_equal"],
        },
        outputs=[
            str(RANKINGS_PATH.relative_to(ROOT)),
            str(METRICS_PATH.relative_to(ROOT)),
        ],
    )


if __name__ == "__main__":
    main()
