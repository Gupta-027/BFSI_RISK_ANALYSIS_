"""
Upload-related UI helpers for the Batch Prediction page.

These only render widgets and small info boxes; all file reading lives in
src/file_processing and all validation in src/validation.
"""

import os

import streamlit as st


def required_columns_info(required_columns):
    """Show the columns the selected module expects."""
    st.info(
        "**Required columns** for this module:\n\n"
        + "  ".join(f"`{col}`" for col in required_columns)
    )


def upload_widget():
    """Render the file uploader and return the uploaded file (or None)."""
    return st.file_uploader(
        "Upload a CSV, Excel or Word file",
        type=["csv", "xlsx", "xls", "docx"],
        help="The file must contain the required columns shown above.",
    )


def sample_download_button(module_label, sample_path):
    """Offer the module's template file so users can fill it and re-upload."""
    if not os.path.exists(sample_path):
        return
    with open(sample_path, "rb") as f:
        st.download_button(
            f"⬇ Download sample input file ({module_label})",
            data=f.read(),
            file_name=os.path.basename(sample_path),
            mime="text/csv",
            use_container_width=True,
        )
