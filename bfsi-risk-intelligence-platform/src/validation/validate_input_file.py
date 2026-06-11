"""
Simple validation helpers for uploaded batch files.

Kept deliberately small: the page decides how to display the result.
"""


def validate_columns(input_df, required_columns):
    """
    Check that every required column is present in the uploaded data.

    Parameters
    ----------
    input_df : pandas.DataFrame
        The uploaded data.
    required_columns : list[str]
        Columns the selected module needs.

    Returns
    -------
    tuple : (is_valid, missing_columns)
        is_valid is True only when no columns are missing.
    """
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    is_valid = len(missing_columns) == 0
    return is_valid, missing_columns


def is_empty(input_df):
    """Return True if the uploaded file has no rows."""
    return input_df is None or input_df.shape[0] == 0
