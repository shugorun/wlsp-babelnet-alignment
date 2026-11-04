"""Write the sheets on which the pairs were annotated.

Reads data/interim/wlsp.parquet, data/interim/pairs.json and
data/interim/synsets.json and writes, under data/interim/sheets/,

    dataset_a.xlsx                          the 1,003 pairs of Dataset A
    dataset_b_part1.xlsx, dataset_b_part2.xlsx
                                            the 5,578 pairs of Dataset B, the
                                            first and the last 100 records

One row per pair: the record ID, the WLSP cell, an empty label cell, the
synset ID, the cells that describe the synset, and the query type that
retrieved it. The two datasets were annotated months apart and their sheets
describe a synset differently (src/wlsp_bn/cells.py).

    uv run python scripts/dataset/07_make_annotation_sheets.py
"""

import json

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

from wlsp_bn import paths
from wlsp_bn.cells import record_text, synset_cells, synset_text_a

WLSP = paths.INTERIM / "wlsp.parquet"
PAIRS = paths.INTERIM / "pairs.json"
SYNSETS = paths.INTERIM / "synsets.json"
OUT = paths.INTERIM / "sheets"
B_PART_SIZE = 100

WIDTHS = {"レコードID番号": 14, "ラベル": 10, "BabelNet番号": 18, "from": 16}
TEXT_WIDTH = 40


def write_sheet(rows: pd.DataFrame, path) -> None:
    """Write the rows to one sheet, with the text cells wrapped."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(list(rows.columns))
    for row in rows.itertuples(index=False, name=None):
        ws.append(list(row))
    for i, name in enumerate(rows.columns, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = WIDTHS.get(
            name, TEXT_WIDTH
        )
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                wrap_text=isinstance(cell.value, str), vertical="top"
            )
    ws.freeze_panes = "A2"
    wb.save(path)


def sheet_rows(
    dataset: str, pairs: list, records: pd.DataFrame, synsets: dict
) -> pd.DataFrame:
    rows = []
    for record_id, sid, query_type in pairs:
        row = {
            "レコードID番号": record_id,
            "分類語彙表情報": record_text(records.loc[record_id]),
            "ラベル": "",
            "BabelNet番号": sid,
        }
        if dataset == "a":
            row["BabelNet情報"] = synset_text_a(synsets, sid)
        else:
            row |= synset_cells(synsets, sid)
        row["from"] = query_type
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    records = pd.read_parquet(WLSP)
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    synsets = json.loads(SYNSETS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    write_sheet(sheet_rows("a", pairs["a"], records, synsets), OUT / "dataset_a.xlsx")
    b = sheet_rows("b", pairs["b"], records, synsets)
    first_part = list(dict.fromkeys(b["レコードID番号"]))[:B_PART_SIZE]
    in_first = b["レコードID番号"].isin(first_part)
    write_sheet(b[in_first], OUT / "dataset_b_part1.xlsx")
    write_sheet(b[~in_first], OUT / "dataset_b_part2.xlsx")
    print(f"{OUT}: sheets of {len(pairs['a']):,} + {len(b):,} pairs")


if __name__ == "__main__":
    main()
