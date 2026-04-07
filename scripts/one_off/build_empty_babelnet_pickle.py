from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"


def build_empty_babelnet_pickle() -> None:
    """Create and save an empty BabelNet table as a pickle file."""
    columns = [
        "hypernym_sids",
        "categories_JA",
        "categories_EN",
        "lemmas_JA",
        "lemmas_EN",
        "glosses_JA",
        "glosses_EN",
        "main_gloss_JA",
        "main_gloss_EN",
    ]

    # Create an empty table with the expected columns.
    df = pd.DataFrame(columns=columns)

    # Keep the intended index name for downstream use.
    df.index.name = "synset_id"

    # Save the empty table.
    df.to_pickle(PROCESSED_DIR / "babelnet_.pkl")


if __name__ == "__main__":
    build_empty_babelnet_pickle()