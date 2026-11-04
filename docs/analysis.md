# Analyses

These are the analyses of the paper's Sections 6.2, 7 and Appendix C. The
scripts in `scripts/analysis/` read the runs that `docs/evaluation.md`
lists, and the comparisons of Section 6.2 and Appendix C are made with
`scripts/evaluation/03_score.py`.

| Result | Command |
|---|---|
| Section 6.2 | `03_score.py ... --compare`, below |
| Table 4, top; Section 7.1, false positives and surface match | `uv run python scripts/analysis/false_positives.py` |
| Table 4, bottom; Section 7.1, error types and shared FNs | `uv run python scripts/analysis/error_types.py` |
| Table 5; Section 7.2 | `uv run python scripts/analysis/dataset_comparison.py` |
| Table 6 (Appendix C) | `03_score.py ... --compare`, below |

The three scripts use Qwen without reasoning tokens, and a majority vote
counts a pair EQUAL when more than half of the runs do.

## False positives

`false_positives.py` gives the false-positive rate of each gold label: the
share of its pairs that a run predicts EQUAL, averaged over the runs of the
condition (two runs of `v9b` and four of `v0` on the development set,
three of each on the test set). It also gives that rate on the test set for
the OT pairs with and without a surface match, a pair whose synset has the
WLSP headword or a co-classified word among its Japanese lemmas, as the
texts given to the models show them.

## Error types

`error_types.py` takes the errors of the majority vote over the three test
runs of each prompt and their types, one per error, from
`data/annotations/error_types_v9b.tsv` and `error_types_v0.tsv`. Each row
of these files is an error: `record_id`, `synset_id`, `error` (FN or FP)
and `type`, one of the types of Appendix B:

| `type` | Appendix B |
|---|---|
| `overlooked_paraphrase` | Overlooked paraphrase |
| `over_applied_rejection_rule` | Over-applied rejection rule |
| `limited_candidate_information` | Limited candidate information |
| `close_synset_confusion` | Confusion between semantically close synsets |
| `overlooked_granularity_difference` | Overlooked granularity difference |
| `category_confusion` | Category confusion |

The first two are the boundary reproduction failures of Table 4. The share
of errors about semantic boundaries counts every type but limited candidate
information and category confusion.

## Datasets A and B

`dataset_comparison.py` compares Dataset A with the candidates of Dataset B
from the same two query types, headword and JMdict, judged on their own,
with the development and test sets combined.

## Comparisons between conditions

The difference between the optimized and the initial prompt on the test
set (Section 6.2), for Qwen and GPT:

```
uv run python scripts/evaluation/03_score.py data/predictions/b-test/qwen3.5-27b/v9b \
    data/predictions/b-test/qwen3.5-27b/v0 --compare
uv run python scripts/evaluation/03_score.py data/predictions/b-test/gpt-5.4-2026-03-05/v9b \
    data/predictions/b-test/gpt-5.4-2026-03-05/v0 --compare
```

Table 6 compares reasoning on and off for each model and prompt, and the
best configuration with reasoning against the initial prompt without it:

```
uv run python scripts/evaluation/03_score.py data/predictions/b-test/qwen3.5-27b/v0-thinking-t0.6 \
    data/predictions/b-test/qwen3.5-27b/v0-t0.6 --compare
uv run python scripts/evaluation/03_score.py data/predictions/b-test/qwen3.5-27b/v9b-thinking-t0.6 \
    data/predictions/b-test/qwen3.5-27b/v9b-t0.6 \
    data/predictions/b-test/qwen3.5-27b/v0-t0.6 --compare
uv run python scripts/evaluation/03_score.py data/predictions/b-test/gpt-5.4-2026-03-05/v0-high \
    data/predictions/b-test/gpt-5.4-2026-03-05/v0 --compare
uv run python scripts/evaluation/03_score.py data/predictions/b-test/gpt-5.4-2026-03-05/v9b-high \
    data/predictions/b-test/gpt-5.4-2026-03-05/v9b \
    data/predictions/b-test/gpt-5.4-2026-03-05/v0 --compare
```
