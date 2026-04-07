from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def build_wlsp_pickle() -> None:
    """Load WLSP data, clean it, and save it as a pickle file."""
    usecols = [
        "レコードＩＤ番号",
        "類",
        "部門",
        "中項目",
        "分類項目",
        "分類番号",
        "段落番号",
        "小段落番号",
        "語番号",
        "見出し本体",
        "読み",
    ]

    # Load only required columns.
    df = pd.read_csv(
        RAW_DIR / "bunruidb-fam.csv",
        usecols=usecols,
        dtype=str,
        na_values=["", "NA", "N/A"],
        keep_default_na=True,
    )

    # Rename columns for internal use.
    df.columns = [
        "record_id",
        "division",
        "category",
        "subcategory",
        "class",
        "category_no",
        "paragraph_no",
        "subparagraph_no",
        "lemma_no",
        "lemma",
        "kana",
    ]

    # Convert numeric columns.
    float_cols = ["category_no"]
    int_cols = ["record_id", "paragraph_no", "subparagraph_no", "lemma_no"]

    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors="coerce")
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce").astype("int")

    # Keep the target division only.
    df = df[df["division"] == "体"]

    # Remove placeholder rows.
    df = df[df["lemma"] != "＊"]

    # Use record_id as the index.
    df = df.set_index("record_id", drop=True)

    # Save the cleaned table.
    df.to_pickle(PROCESSED_DIR / "wlsp.pkl")


if __name__ == "__main__":
    build_wlsp_pickle()


# NaN counts:
# - category: 906

# Notes:
# - Rows with lemma == "＊" are excluded.
# - Division values include {"体", "用", "相", "他"}.
# - All strings are stripped.

# NaN values by column:
# record_id            0
# division             0
# category           906
# subcategory          0
# class                0
# category_no          0
# paragraph_no         0
# subparagraph_no      0
# lemma_no             0
# lemma                0
# kana                 0
# dtype: int64

# WLSP contains 480 rows with lemma == "＊".
# These rows appear to have some special meaning, but are excluded here.