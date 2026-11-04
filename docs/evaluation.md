# Evaluation

This is how the LLMs were evaluated (paper Section 5) and how the runs
behind the results in Sections 6 and 7 and Appendix C are made. The scripts
are in `scripts/evaluation/` and read the model inputs that
`scripts/dataset/09_build_model_inputs.py` writes.

| Step | Script | Writes |
|---|---|---|
| 1 | `01_split.py` | `data/splits/dataset_b.tsv`: the development and test records of Dataset B |
| 2 | `02_judge.py` | `data/predictions/<data>/<model>/<condition>/run<k>.tsv`: the label of each pair in one run |
| 3 | `03_score.py` | precision, recall and F1 over the runs of each condition, and differences between conditions |

## Split

Dataset B is split at the record level into 100 development and 100 test
records. The split is stratified by WLSP division and by the number of EQ
pairs of a record, and among the stratified splits that seeds 0 to 9,999
give, the one kept is the most balanced in its numbers of pairs, of EQ, BR
and NR pairs, and of pairs with a surface match (`01_split.py` gives the
details). The prompts were chosen on the development set; Dataset A is not
split.

## Judging

Each request holds one WLSP record and a chunk of its candidates, and the
model labels each candidate EQUAL or NONE; EQUAL is the positive class. The
prompts and how they make their requests are in
[`prompts/alignment/`](../prompts/alignment/README.md): `v0` is the initial
prompt and `v9b` the optimized one.

Qwen3.5-27B was served with vLLM on one NVIDIA A100 80GB GPU:

```
docker run --gpus all -p 8000:8000 vllm/vllm-openai:v0.25.1-cu129 \
    --model Qwen/Qwen3.5-27B --served-model-name qwen3.5-27b \
    --gpu-memory-utilization 0.92 --max-num-seqs 32 --reasoning-parser qwen3
```

`02_judge.py` reaches it at `http://localhost:8000/v1` (`--base-url` to
change), with greedy decoding and thinking off unless `--thinking` and
`--temperature` are given; a temperature above 0 samples with top_p 0.95
and top_k 20, the settings recommended for the model. GPT-5.4 is the
snapshot `gpt-5.4-2026-03-05` of the OpenAI Responses API, called with
`--api openai --model gpt-5.4-2026-03-05` and `OPENAI_API_KEY` set, with
reasoning effort none unless `--effort` is given. Each condition is run three times
unless `--runs` says otherwise, with eight requests in parallel.

The runs behind each result:

| Result | Data | Model | Prompts | Options |
|---|---|---|---|---|
| Section 5.2: prompt search | `--data b --split dev` | Qwen | all 53 | `--runs 1` to `--runs 4` |
| Table 2 | `--data b --split test` | Qwen, GPT | `v0`, `v9b` | |
| Table 3 | `--data a` | Qwen, GPT | `v0`, `v9b` | |
| Table 5 | `--data b`, `--split dev` and `test`, `--query-types headword jmdict` | Qwen | `v0` | |
| Table 6 (Appendix C) | `--data b --split test` | Qwen | `v0`, `v9b` | `--temperature 0.6`, and `--thinking --temperature 0.6 --runs 1` |
| Table 6 (Appendix C) | `--data b --split test` | GPT | `v0`, `v9b` | `--effort high` |

For example, the Qwen rows of Table 2 and the difference between the two
prompts:

```
uv run python scripts/evaluation/02_judge.py --data b --split test --prompt v0 --model qwen3.5-27b
uv run python scripts/evaluation/02_judge.py --data b --split test --prompt v9b --model qwen3.5-27b
uv run python scripts/evaluation/03_score.py data/predictions/b-test/qwen3.5-27b/v9b \
    data/predictions/b-test/qwen3.5-27b/v0 --compare
```

## Scoring

Precision, recall and F1 are those of the EQ class, averaged over the runs
of a condition, with the sample standard deviation of F1 across runs. A
difference between two conditions is that of their mean F1, with a 95%
confidence interval from 10,000 bootstrap replicates that resample records,
since the pairs of one record are not independent. The baselines of Table 2
write their predictions in the same form, as one run each
([docs/baselines.md](baselines.md)), so they are scored and compared in the
same way.
