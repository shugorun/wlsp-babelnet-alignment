"""Score the runs of one or more conditions against the annotations.

Each argument is a condition directory written by 02_judge.py, such as
data/predictions/b-test/qwen3.5-27b/v0. EQUAL predictions are scored
against the EQ labels of data/annotations/. For each condition the script
prints the precision, recall and F1 averaged over its runs, and the sample
standard deviation of F1 across them.

With --compare, it also prints the difference in mean F1 between the first
condition and each of the others, with its 95% confidence interval from a
bootstrap over records (src/wlsp_bn/metrics.py). The conditions compared
must cover the same pairs.

    uv run python scripts/evaluation/03_score.py \\
        data/predictions/b-test/qwen3.5-27b/v0 \\
        data/predictions/b-test/qwen3.5-27b/v9b --compare
"""

import argparse
import statistics
from pathlib import Path

from wlsp_bn.metrics import mean_f1_diff_ci, precision_recall_f1
from wlsp_bn.predictions import KEY, load_runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("conditions", nargs="+", type=Path)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    loaded = [load_runs(c) for c in args.conditions]
    width = max(len(str(c)) for c in args.conditions)
    print(f"{'condition':{width}}  runs      P      R     F1     SD")
    for condition, (pairs, runs) in zip(args.conditions, loaded, strict=True):
        y = (pairs["label"] == "EQ").tolist()
        scores = [precision_recall_f1(y, run) for run in runs]
        p, r, f1 = (statistics.mean(s[i] for s in scores) for i in range(3))
        sd = f"{statistics.stdev(s[2] for s in scores):.3f}" if len(runs) > 1 else "-"
        print(
            f"{condition!s:{width}}  {len(runs):4}  {p:.3f}  {r:.3f}  {f1:.3f}  {sd:>5}"
        )

    if args.compare:
        base_pairs, base_runs = loaded[0]
        y = (base_pairs["label"] == "EQ").tolist()
        for condition, (pairs, runs) in zip(
            args.conditions[1:], loaded[1:], strict=True
        ):
            if not pairs[KEY].equals(base_pairs[KEY]):
                raise SystemExit(f"{condition} does not cover the same pairs")
            diff, lo, hi = mean_f1_diff_ci(
                base_pairs["record_id"].tolist(), y, base_runs, runs
            )
            ci = f"[{lo:+.3f}, {hi:+.3f}]"
            print(f"{args.conditions[0]} - {condition}: {diff:+.3f} {ci}")


if __name__ == "__main__":
    main()
