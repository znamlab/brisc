"""Reading and writing of Source Data Excel workbooks."""

from pathlib import Path

import numpy as np
import pandas as pd


def save_excel_sheets(panels_dict, output_path, notes_dict=None, expected=None):
    """
    Save a dictionary of DataFrames to an Excel workbook, one sheet per key.

    Args:
        panels_dict (dict): Dictionary mapping sheet names (str) to pandas DataFrames.
        output_path (str or Path): Path to output Excel file.
        notes_dict (dict, optional): Dictionary mapping sheet names to note strings.
        expected (list, optional): Every sheet the figure should contain. Any that are
            absent are reported loudly, so a renamed or unpassed variable cannot make a
            panel disappear silently.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if expected:
        missing = [name for name in expected if name not in panels_dict]
        if missing:
            print(
                f"[Source Data] !! MISSING {len(missing)}/{len(expected)} panels for "
                f"{output_path.name}: {missing}"
            )

    if not panels_dict:
        print(f"[Source Data Warning] No panels to save for {output_path}")
        return output_path

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in panels_dict.items():
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            # Excel sheet names max 31 chars
            clean_name = str(sheet_name)[:31].replace(":", "_").replace("/", "_")
            note = notes_dict.get(sheet_name) if notes_dict else None
            startrow = 2 if note else 0
            df.to_excel(writer, sheet_name=clean_name, startrow=startrow, index=False)
            if note:
                ws = writer.sheets[clean_name]
                ws.cell(row=1, column=1, value=note)

    print(
        f"[Source Data] Saved {output_path} with {len(panels_dict)} sheets: "
        f"{list(panels_dict.keys())}"
    )
    return output_path


def read_source_data_workbook(xlsx_path):
    """Read back a Source Data workbook written by :func:`save_excel_sheets`.

    Sheets that carry a note have the note in cell A1 and the table starting at row 3
    (``startrow=2``), so the header row cannot be assumed to be the first one. Each
    sheet is first read without a header to locate the first row holding more than one
    non-null cell, which is then used as the header.

    Args:
        xlsx_path (str or Path): Path to the workbook.

    Returns:
        dict: Mapping of sheet name to DataFrame.
    """
    xlsx_path = Path(xlsx_path)
    sheets = {}
    with pd.ExcelFile(xlsx_path, engine="openpyxl") as xls:
        for sheet_name in xls.sheet_names:
            raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            if raw.empty:
                sheets[sheet_name] = pd.DataFrame()
                continue
            header_row = 0
            for i in range(min(len(raw), 10)):
                if raw.iloc[i].notna().sum() > 1:
                    header_row = i
                    break
            df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=header_row)
            sheets[sheet_name] = df
    return sheets


def extract_counts_array(lib_data):
    if isinstance(lib_data, np.ndarray):
        if lib_data.ndim == 2 and lib_data.shape[1] >= 2:
            return lib_data[:, 1]
        elif lib_data.ndim == 1:
            return lib_data
    elif isinstance(lib_data, pd.DataFrame) and "umi_count" in lib_data.columns:
        return lib_data["umi_count"].values
    return None
