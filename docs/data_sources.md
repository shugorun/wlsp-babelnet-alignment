# Data sources

The scripts need three external resources: WLSP, JMdict and BabelNet. None
of them can be redistributed in this repository, so each user obtains them.
The locations of WLSP and JMdict go in `configs/paths.toml` (copied from
`configs/paths.example.toml`); BabelNet is given to its RPC server
([docs/dataset_construction.md](dataset_construction.md)).

| Key in `paths.toml` | Resource | Version used | License |
|---|---|---|---|
| `wlsp_csv` | WLSP database, revised and enlarged edition, with word familiarity ratings (`bunruidb-fam.csv`), from [WLSP-familiarity](https://github.com/masayu-a/WLSP-familiarity) of the National Institute for Japanese Language and Linguistics | 4.0 | CC BY-NC-SA 3.0 |
| `jmdict_xml` | JMdict, English edition (`JMdict_e.xml`), from the [Electronic Dictionary Research and Development Group](https://www.edrdg.org/jmdict/j_jmdict.html) | created 2025-11-02 | CC BY-SA |
| — | BabelNet 5.0 offline index, from [Babelscape](https://babelnet.org) | 5.0 | BabelNet Non-Commercial License |

The JMdict file used had 61,217,565 bytes and the SHA-256
`d74eb9652b4d02509de21894a4cdf278c28e392f90cf59c79e2aa3851fc184b6`.

## What is published here

- The code, the prompts and the output schemas.
- `data/term_expansion/dataset_b.jsonl`: the aliases and spelling variants
  that GPT-5.4 generated for the 300 sampled records of Dataset B.

Nothing that holds BabelNet or WLSP content is published: the scripts write
it to `data/interim/`, which git ignores.

Not included are the annotations, that is the labels of the two datasets,
the two corrections of Dataset A and the error types of Section 7.1, in
`data/annotations/`, and the predictions of the LLM runs and the baselines,
in `data/predictions/`. The scripts from step 8 of the dataset construction
on, the scoring, the baselines and the analyses read them from there.
