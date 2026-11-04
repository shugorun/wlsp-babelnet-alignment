"""The types of the errors of Qwen on the test set of Dataset B (Table 4,
bottom, and Section 7.1).

The errors are those of the majority vote over the three runs of the
optimized (v9b) and the initial (v0) prompt, without reasoning tokens. The
type of each error (Appendix B) is in data/annotations/error_types_<prompt>.tsv,
which must cover exactly these errors. The script prints the share of each
type among the FNs and among the FPs; the share of all errors whose type
concerns reproducing semantic boundaries, that is all types but limited
candidate information and category confusion; and the FNs the two prompts
share, with their Jaccard coefficient.

    uv run python scripts/analysis/error_types.py
"""

import pandas as pd

from wlsp_bn import paths
from wlsp_bn.predictions import KEY, load_runs, majority

PROMPTS = {"optimized": "v9b", "initial": "v0"}
# The rows of Table 4: each groups one or more types of Appendix B.
ROWS = {
    "FN": {
        "Boundary reproduction failure": [
            "overlooked_paraphrase",
            "over_applied_rejection_rule",
        ],
        "Limited candidate information": ["limited_candidate_information"],
    },
    "FP": {
        "Confusion between semantically close synsets": ["close_synset_confusion"],
        "Overlooked granularity difference": ["overlooked_granularity_difference"],
        "Category confusion": ["category_confusion"],
    },
}
NOT_BOUNDARY = {"limited_candidate_information", "category_confusion"}


def errors(prompt: str) -> pd.DataFrame:
    """The pairs of the majority vote's errors, with FN or FP."""
    pairs, runs = load_runs(paths.PREDICTIONS / "b-test" / "qwen3.5-27b" / prompt)
    eq = pairs["label"] == "EQ"
    predicted = pd.Series(majority(runs))
    wrong = pairs[eq != predicted].copy()
    wrong["error"] = eq[eq != predicted].map({True: "FN", False: "FP"})
    return wrong[[*KEY, "error"]]


def main() -> None:
    typed = {}
    for name, prompt in PROMPTS.items():
        types = pd.read_csv(paths.ANNOTATIONS / f"error_types_{prompt}.tsv", sep="\t")
        merged = errors(prompt).merge(
            types, on=KEY, how="outer", validate="one_to_one", indicator=True
        )
        if (merged["_merge"] != "both").any() or (
            merged["error_x"] != merged["error_y"]
        ).any():
            raise SystemExit(
                f"error_types_{prompt}.tsv does not cover the errors of {prompt}"
            )
        typed[name] = merged.rename(columns={"error_x": "error"})

    print(f"{'':48}" + "".join(f"{n:>12}" for n in PROMPTS))
    for error, rows in ROWS.items():
        counts = {n: (t["error"] == error).sum() for n, t in typed.items()}
        print(f"{error} ({', '.join(f'{n} {c}' for n, c in counts.items())})")
        for row, members in rows.items():
            shares = [
                t["type"].isin(members).sum() / counts[n] for n, t in typed.items()
            ]
            print(f"  {row:46}" + "".join(f"{s:12.0%}" for s in shares))

    print("\nShare of all errors about semantic boundaries")
    for name, t in typed.items():
        boundary = ~t["type"].isin(NOT_BOUNDARY)
        print(f"  {name}: {boundary.sum()}/{len(t)} = {boundary.mean():.0%}")

    fns = {
        n: set(map(tuple, t.loc[t["error"] == "FN", KEY].to_numpy()))
        for n, t in typed.items()
    }
    shared = set.intersection(*fns.values())
    union = set.union(*fns.values())
    print(
        f"\nFNs shared by the two prompts: {len(shared)}, "
        f"Jaccard {len(shared) / len(union):.2f}"
    )


if __name__ == "__main__":
    main()
