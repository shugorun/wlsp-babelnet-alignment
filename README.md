# WLSP–BabelNet Alignment

Code for the PACLIC 40 (2026) paper *Aligning the Japanese Word List by
Semantic Principles with BabelNet: Challenges in Automatic Lexical Resource
Alignment*.

The code of the earlier NLP2026 paper is kept under the
[`nlp2026` release](https://github.com/shugorun/wlsp-babelnet-alignment/releases/tag/nlp2026).

## Setup

[uv](https://docs.astral.sh/uv/) is needed.

```
uv sync --locked
```

Copy `configs/paths.example.toml` to `configs/paths.toml` and write where
the external resources are on your machine; [docs/data_sources.md](docs/data_sources.md)
says where to obtain them and what this repository includes. The
annotations of the datasets and the predictions of the models are not
included. The baselines that use E5 also need
`uv sync --locked --extra baselines`.

## Contents

| Paper | Code | Description |
|---|---|---|
| Section 3 | `scripts/dataset/` | Construction of Datasets A and B: [docs/dataset_construction.md](docs/dataset_construction.md) |
| Section 5 | `scripts/evaluation/` | Split of Dataset B, judging with the LLMs and scoring: [docs/evaluation.md](docs/evaluation.md) |
| Section 5.2, Appendix A | `prompts/alignment/` | The 53 prompts of the prompt search: [prompts/alignment/README.md](prompts/alignment/README.md) |
| Section 5.3 | `scripts/baselines/` | The baselines of Table 2: [docs/baselines.md](docs/baselines.md) |
| Sections 6.2 and 7, Appendix C | `scripts/analysis/` | The analyses of the results: [docs/analysis.md](docs/analysis.md) |

## License

The code is released under the MIT License (see [LICENSE](LICENSE)).
