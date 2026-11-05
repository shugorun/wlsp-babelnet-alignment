"""All-positive: every pair of the test set is predicted EQ.

Its F1 is the lowest a method can reach by predicting more pairs EQ:
2p / (1 + p) for the share p of EQ pairs.

    uv run python scripts/baselines/all_positive.py
"""

import numpy as np

from wlsp_bn import baselines as bl


def main() -> None:
    test = bl.load_split("test")
    bl.write_predictions("all-positive", test, np.ones(len(test), bool), {})


if __name__ == "__main__":
    main()
