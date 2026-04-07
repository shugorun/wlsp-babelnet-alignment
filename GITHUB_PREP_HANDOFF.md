# GitHub Prep Handoff

## Current Goal

Prepare this repository for public GitHub release as a job-hunting portfolio repository.

Main requirements already discussed:

- Hide or remove private paths, secrets, and non-public artifacts.
- Keep the repository structure clean and readable.
- Add enough comments to important programs so the implementation intent is understandable.
- Create a root `README.md` that explains the research overview and links to more detailed READMEs.
- Organize destination READMEs for each major method area.

No cleanup work has been done yet in this phase. This file only records the current state and the recommended order of work.

## Current Baseline State

The working repo is now this repository root.

Baseline-related scripts are organized under:

- `src/baselines/`

Baseline-related docs are under:

- `docs/baselines/`

### gold_B reproducible sequence

The clean rerun sequence is:

1. `src/baselines/run_gold_b_baseline.py`
2. `src/baselines/run_gold_b_ranking.py`
3. `src/baselines/run_gold_b_hybrid_search.py`
4. `src/baselines/run_gold_b_record_decoder.py`

### gold_B results

Reproducible results currently adopted:

- Pairwise lexical baseline:
  - pairwise `EQUAL` F1 = `0.545918`
- E5 ranking baseline best:
  - `natural + ja_preferred + top1`
  - pairwise `EQUAL` F1 = `0.657459`
- Hybrid best:
  - config stored at `outputs/baselines/gold_B_hybrid_best/config.json`
  - pairwise `EQUAL` F1 = `0.684466`
- Record decoder best:
  - record exact match rate = `0.491525`

Important note:

- An older summary mentioned hybrid `0.682171`, but that exact condition was not reproducible from preserved artifacts.
- The clean rerun produced a reproducible hybrid best of `0.684466`, which is now the reliable best result.

### Main baseline artifacts

- `outputs/baselines/gold_B/`
- `outputs/baselines/gold_B_ranking/`
- `outputs/baselines/gold_B_hybrid_search/`
- `outputs/baselines/gold_B_hybrid_best/`
- `outputs/baselines/gold_B_record_decoder/`

### Baseline logs and summaries

- `docs/baselines/EXPERIMENT_LOG.md`
- `docs/baselines/gold_B_baseline_summary.md`
- `docs/baselines/gold_B_baseline_summary_legacy.md`

## gold_A state

### gold_A using best gold_B hybrid

Script:

- `src/baselines/run_gold_a_hybrid_best.py`

This applies the best reproducible `gold_B` hybrid configuration to `gold_A` with the same feature columns, threshold, and ranking setup.

Current result:

- pairwise `EQUAL` F1 = `0.5319148936170213`
- record macro F1 = `0.4569767441860465`
- record exact match rate = `0.3023255813953488`

Outputs:

- `outputs/baselines/gold_A_hybrid_best/config.json`
- `outputs/baselines/gold_A_hybrid_best/summary.json`
- `outputs/baselines/gold_A_hybrid_best/predictions.pkl`
- `outputs/baselines/gold_A_hybrid_best/pair_predictions.csv`

### gold_A methods that do not learn from gold_B

Script:

- `src/baselines/run_gold_a_no_training_methods.py`

This compares methods that do not train on `gold_B`:

- weighted lexical ranking
- E5 ranking
- MPNet ranking
- pretrained cross-encoder

Current best-per-method results on `gold_A`:

- `cross_encoder_pretrained`
  - `concise + ja_preferred + top1`
  - pairwise `EQUAL` F1 = `0.5512820512820513`
- `e5_ranking`
  - `headword_synonyms__sentence_gloss_lemmas + ja_preferred + top2`
  - pairwise `EQUAL` F1 = `0.54`
- `mpnet_ranking`
  - `concise + ja_preferred + top3`
  - pairwise `EQUAL` F1 = `0.48888888888888893`
- `weighted_lexical`
  - `weighted_score + ja_preferred + top2`
  - pairwise `EQUAL` F1 = `0.48`

Outputs:

- `outputs/baselines/gold_A_no_training_methods/summary.csv`
- `outputs/baselines/gold_A_no_training_methods/summary.pkl`
- plus per-method ranking files in the same directory.

Interpretation:

- On `gold_A`, the best non-training method currently is the pretrained cross-encoder.
- It slightly outperforms direct transfer of the best `gold_B` hybrid.

## Things already identified as necessary before GitHub release

This section started as a todo list. Some items below are now partially done, so treat it as a progress memo rather than a pure pending list.

### 1. Hide or remove private / non-public content

Must inspect and remove or hide:

- API keys or key files
- absolute local paths
- environment-specific local paths
- private data or raw data that should not be published
- outputs that should not be committed
- temporary or trash files that make the repo look messy

Current status:

- `.gitignore` has been tightened
- absolute local paths have been abstracted in the main BabelNet launch paths
- this area still needs a final public-file check before publishing

### 2. Decide the public-facing repository structure

Need to decide what stays visible and what should be removed or ignored.

Especially review:

- `trash/`
- large `outputs/`
- `data/`
- experimental leftovers
- old scripts outside `src/baselines/`

### 3. Improve code readability

Need to add or revise comments in important scripts, especially:

- `src/baselines/run_gold_b_baseline.py`
- `src/baselines/run_gold_b_ranking.py`
- `src/baselines/run_gold_b_hybrid_search.py`
- `src/baselines/run_gold_b_record_decoder.py`
- `src/baselines/run_gold_a_hybrid_best.py`
- `src/baselines/run_gold_a_no_training_methods.py`

Goal:

- not excessive comments
- enough comments to explain what each stage is doing and why
- style should stay consistent with the rest of the repo

### 4. Build README structure

Need:

- root `README.md`
- destination READMEs for detailed method areas

Root README should include:

- research overview
- task definition
- data / candidate setup at a high level
- non-API baselines summary
- API alignment summary
- directory guide
- links to detailed docs

Likely destination READMEs:

- `docs/baselines/README.md`
- maybe one for API alignment side
- maybe one for data / outputs handling if needed

Current status:

- root `README.md` exists
- `docs/baselines/README.md` exists
- `docs/alignment/README.md` exists
- `data/README.md` exists
- `outputs/README.md` exists

### 5. Clarify reproducibility

Need to explain clearly:

- which results are the current reproducible best
- which older values are legacy notes only
- what can be reproduced from the public repo
- what cannot be reproduced due to private data or environment limits

Current status:

- reproducible best vs legacy best is already documented in `docs/baselines/`
- the public README now explains that the repository is code-and-summary centric, not a full data mirror

## Recommended Order of Work

This is the recommended sequence for the next phase.

1. Inventory all files that should not be public.
2. Decide which directories and outputs remain in the public repo.
3. Remove or hide secrets, private paths, and noisy artifacts.
4. Clean up comments in the main scripts.
5. Build the root `README.md`.
6. Build the destination READMEs and link them from the root README.
7. Final pass for path hygiene, broken links, and repo readability.

## Important Constraint

When continuing from this handoff, avoid changing unrelated research code unless necessary for publication hygiene.

The next requested phase is about:

- publication cleanup
- comments
- README organization
- path / secret masking

and not about further model experimentation.
