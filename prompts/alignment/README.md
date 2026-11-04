# Alignment prompts

The 53 prompts compared on the development set of Dataset B (paper
Section 5.2): the initial prompt `v0`, fixed before the search, and the 52
configurations of the search. The optimized prompt is `v9b`. Each
directory holds the instructions (`system.txt`), the JSON schema of the
answer (`schema.json`) and how the requests are made (`config.json`; see
`src/wlsp_bn/alignment.py`). `v4hv`, `v5av` and `v7av` are the second-pass
prompts of `v4h`, `v5a` and `v7a`.

Unless the table says otherwise, a prompt gives the full record text, the
model labels each candidate independently, and the candidates are given in
the order of the data.

| Prompt | Aspect | Change |
|---|---|---|
| `v0` | | The initial prompt: EQUAL for a synset with the same meaning at the same granularity, NONE otherwise. 25 candidates per request, with the full synset text (`bn_full`). |
| `v1a` | Decision criteria | Equivalence of concepts, the record's concept being defined by its headword, co-classified words and category path; a lemma mismatch or the lack of a Japanese lemma is no reason for NONE; several candidates can be EQUAL. |
| `v1b` | Decision criteria | `v1a` with four conditions for NONE: broader, narrower, a shift to a part, instrument or attribute, and a different sense of the same word. |
| `v1c` | Decision criteria | EQUAL by default for a natural name of the concept; NONE only under specific conditions. |
| `v1d` | Decision criteria | EQUAL only when four checks all pass: an anchor, a compatible type, the same granularity, and not a different sense of the same word. |
| `v1e` | Output format | `v1b`, with the core sense of the candidate (at most 12 words) written before its label. |
| `v2a` | Demonstrations | `v1c` with examples of hard negatives (a different sense of the same word and a narrower sense, both NONE) and one EQUAL. |
| `v2b` | Demonstrations | `v1c` with an example of several EQUAL candidates: a paraphrase and an English-only match as EQUAL, a shift of agent as NONE. |
| `v2c` | Demonstrations | `v1c` with one example from each WLSP division, five in all. |
| `v2d` | Demonstrations | `v1c` with the examples of `v2a` and `v2b`. |
| `v2e` | Output format | `v1c` with four labels, EQUAL, HYPERNYM, HYPONYM and NONE; only EQUAL is positive. |
| `v3a` | Input and candidates | `v1c` with the shorter synset text (`bn_1step`: no relations, no IS A targets two steps up, no Chinese). |
| `v3b` | Input and candidates | `v1c` with 10 candidates per request. |
| `v3c` | Input and candidates | `v1c` with the shorter synset text and 10 candidates per request. |
| `v3d` | Input and candidates | `v1c` with the shorter record text (`wlsp_min`: the headword and its reading). |
| `v3e` | Input and candidates, demonstrations | `v3c` with the examples of `v2a`. |
| `v3f` | Input and candidates | `v1c` with the shorter synset text and 50 candidates per request. |
| `v3g` | Input and candidates | `v1c` with 40 candidates per request. |
| `v4a` | Judgment mode | `v3f`, comparing the candidates: the answer lists the EQUAL ones. |
| `v4c` | Instruction structure | `v3f` rewritten as five numbered steps. |
| `v4d` | Instruction structure | `v3f` rewritten as a specification with numbered clauses. |
| `v4e` | Decision criteria | `v3f` with two rules, NONE without evidence and agreement between the category path and the synset's domain, the rest of the text being reworded too. |
| `v4f` | Judgment mode | `v4a` with the list limited to 50 items. |
| `v4h` | Judgment mode | A list of at most 20 candidates, each then checked strictly by `v4hv`. |
| `v5a` | Judgment mode | A list of at most 20 candidates, each then checked by `v5av`, which has the text of `v3f`. |
| `v5b` | Decision criteria | `v3f` with a sixth condition for NONE, the lack of evidence, and a clause that protects clear matches from it. |
| `v5c` | Demonstrations | `v3f` with one invented example of four candidates. |
| `v6a` | Decision criteria | `v5b` with the protecting clause stated as a negation. |
| `v6c` | Decision criteria | `v3f` with two rules for the Relations division: a synset without a gloss is judged by its IS A targets, and senses that are personality traits are excluded. |
| `v6d` | Decision criteria | `v3f` with its first two conditions applying only when the level really differs. |
| `v7a` | Judgment mode | A list that must copy the candidate IDs exactly, each listed candidate then checked leniently by `v7av`. |
| `v8a` | Demonstrations | `v3f` with two invented examples showing several EQUAL candidates, four EQUAL and three NONE, each NONE naming its condition. |
| `v9a` | Input and candidates | `v3f` with 75 candidates per request. |
| `v9b` | Decision criteria | `v3f`, stating that some records have no equivalent in BabelNet, so that NONE for every candidate can be right. The optimized prompt. |
| `v9c` | Decision criteria | `v3f`, defining the record as a semantic class and no longer asking for synonymy. |
| `v9e` | Judgment mode | `v4a` with at most three listed candidates, stating that an empty list can be right. |
| `v9f` | Input and candidates | `v5b` with 75 candidates per request. |
| `v10a` | Decision criteria | `v9b`, stating that a synset need not be a synonym of the headword: any meaning within the range of the co-classified words is enough. |
| `v10b` | Decision criteria | `v9b`, adding not to exclude a fitting candidate nor read the conditions more strictly than written. |
| `v10c` | Decision criteria | `v9b`, adding that one EQUAL candidate does not settle a record and the others are judged alike. |
| `v11a` | Output format | `v9b` with the number of the condition that applies written before each label, and EQUAL when none can be named. |
| `v11b` | Output format | `v9b` with the core sense of the candidate (at most 12 words) written before its label. |
| `v12a` | Output format | `v9b` with the number of the condition that applies written before each label. |
| `v13a`–`v13d` | Instruction structure | Rewordings of `v9b` with the same content and structure. |
| `v14a` | Input and candidates | `v9b` with the candidates in reverse order. |
| `v14b` | Input and candidates | `v9b` with the candidates shuffled. |
| `v14c` | Input and candidates | `v3f` with the candidates in reverse order. |
| `v15b` | Instruction structure | `v9b` written in Japanese. |
| `v15c` | Instruction structure | `v9b` opening with a lexicographer persona. |
| `v16a` | Instruction structure | `v3f` opening with the persona of `v15c`. |
