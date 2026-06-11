"""
Read an uploaded data file into a pandas DataFrame.

Supports CSV, Excel (.xlsx / .xls) and, optionally, a simple Word table
(.docx). The Word path is delegated to read_word_table so this file stays
focused on the common spreadsheet formats.
"""

import pandas as pd


def read_uploaded_file(uploaded_file):
    """
    Detect the file type from its name and return a DataFrame.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile (or any file-like object with a
        ``.name`` attribute).

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith((".xlsx", ".xls")):
        # Requires openpyxl (declared in requirements.txt).
        return pd.read_excel(uploaded_file)

    if name.endswith(".docx"):
        # Imported lazily so python-docx is only needed if a Word file is used.
        from src.file_processing.read_word_table import read_word_table
        return read_word_table(uploaded_file)

    raise ValueError(
        "Unsupported file type. Please upload a CSV, Excel (.xlsx/.xls) "
        "or Word (.docx) file."
    )
