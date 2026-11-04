# Dataset construction

This is how the two datasets of the paper (Section 3) were built: Dataset A
(86 records, 1,003 record–synset pairs) and Dataset B (200 records, 5,578
pairs). The scripts are in `scripts/dataset/` and run in the order of their
numbers. Each reads the files the previous ones wrote; everything that
holds WLSP or BabelNet content goes to `data/interim/`, which is not
committed.

| Step | Script | Environment | Writes |
|---|---|---|---|
| 1 | `01_prepare_wlsp.py` | main | `wlsp.parquet`: the 66,884 Nominal records, their co-classified words and JMdict terms |
| 2 | `02_sample_records.py` | main | `samples.json`: 100 records for A, 300 for B |
| 3 | `03_make_queries.py` | main | `queries.json`: the query terms of each record; the generated terms of B in `data/term_expansion/dataset_b.jsonl` |
| 4 | `04_fetch_candidates.py` | BabelNet | `candidates.json`: the synsets each query retrieves |
| 5 | `05_select_pairs.py` | main | `pairs.json`: the pairs of the 86 and 200 records |
| 6 | `06_fetch_synsets.py` | BabelNet | `synsets.json`: what the sheets show of each synset |
| 7 | `07_make_annotation_sheets.py` | main | `sheets/`: the annotation sheets |
| — | (annotation by hand) | | the labelled sheets |
| 8 | `08_import_annotations.py` | main | `data/annotations/dataset_{a,b}.tsv`: the labels |
| 9 | `09_build_model_inputs.py` | main | `pairs_{a,b}.parquet`: the text of each pair given to the models |

## Environments

The main environment is the one `uv sync` makes; run a script with
`uv run python scripts/dataset/<script>`. Step 3 calls the OpenAI API only
with `--call-api` and then needs `OPENAI_API_KEY`.

Steps 4 and 6 query the local BabelNet 5.0 index through the BabelNet Python
API 1.2.0, which needs Python 3.8, and its RPC server, which runs in Docker:

```
uv venv --python 3.8 .venv-babelnet
uv pip install --python .venv-babelnet -r scripts/dataset/requirements-babelnet.txt
```

With that environment activated (`source .venv-babelnet/bin/activate`, or
`.venv-babelnet\Scripts\activate` on Windows), start the RPC server on the
index and leave it running, then run the two steps with `python`:

```
babelnet-rpc start --bn <path to the BabelNet 5.0 index> --mode tcp --tcp 7790 --no-doc
python scripts/dataset/04_fetch_candidates.py
```

The scripts find the server through `configs/babelnet_conf.yml`.

## How each dataset was built

**Records.** Both datasets take records of the Nominal class.
Dataset A takes its records from the Nature division, stratified by
the number of decimal digits of the classification number (1 to 4). The sample was drawn in two
stages: 150 records spread over the four strata, which gave 140 because the
1-digit stratum has only 27 records, and then 25 from each stratum of those
140. Of these 100 records, 14 have no candidate synset and were left out,
which leaves 86. Dataset B is a random sample of 300 records, of which 266
have candidates; 200 of those were selected at random.

**Candidates.** Each query term is searched in BabelNet as a noun in its
language, and only concept synsets are kept, which excludes named entities.
Dataset A uses two query types: the headword, and the English terms JMdict
gives for it. Dataset B adds four: Japanese and English aliases, and
Japanese and English spelling variants, which GPT-5.4 generated from the
headword, its reading, its classification, its JMdict terms and its
co-classified words. A synset retrieved by several query types counts once,
under the first of: headword, JMdict, Japanese alias, English alias,
Japanese variant, English variant.

**Term expansion.** The prompt is in `prompts/term_expansion/system.txt` and
the output schema in `schema.json`. The model was `gpt-5.4-2026-03-05` with
reasoning effort high, 20 records per request.
The prompt also asks for hypernyms; they were not used as queries and are
not kept. Generated terms that repeat the headword or a JMdict term are
dropped, and at most three are kept per field.

**Annotation sheets.** The sheets of Dataset B show each synset's lemmas,
categories and glosses, its relations, its IS A targets two steps up, and
its Chinese lemmas, categories and glosses. The sheet of Dataset A, annotated
months earlier, shows a shorter description: lemmas, categories, hypernyms
and glosses.

**Labels.** The sheets were labelled OK, ⊃, ⊂ and NG, which are the paper's
EQ, BR, NR and OT. Two labels of Dataset A were corrected afterwards, with
the annotator's agreement (`data/annotations/dataset_a_corrections.tsv`):
records 陰 and 陽 against the synset of yin and yang as a whole, a part–whole
pair that the annotation guideline labels OT.

**Model inputs.** Both datasets give the models the text of the sheets of
Dataset B, also for Dataset A.
