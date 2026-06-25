"""
Batch Risk Scoring page.

Lets a BFSI user upload a CSV / Excel / Word file of many records, score every
row with the chosen model, then filter and download the results — like a real
bank/NBFC/fintech risk dashboard.

This file only orchestrates the UI. All real work is delegated:
  - reading files        -> src/file_processing
  - validating columns   -> src/validation
  - scoring rows         -> src/prediction/batch_predictor
  - risk categories      -> src/utils/risk_rules
"""

import streamlit as st

from src.file_processing.read_csv_excel import read_uploaded_file
from src.validation.validate_input_file import validate_columns, is_empty
from src.prediction.batch_predictor import (
    MODULE_NAMES, get_module_config, run_batch_prediction,
)
from frontend.components.file_uploader import (
    required_columns_info, upload_widget, sample_download_button,
)
from frontend.components.data_table import summary_kpis, results_table


def _step(number, title):
    """Render a numbered step heading for the guided flow."""
    st.markdown(
        f"""
        <div class="step-head">
            <span class="step-num">{number}</span>
            <span class="step-text">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.markdown('<div class="section-title">&#128202; Batch Risk Scoring</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Upload a customer or transaction file and '
        'score every record at once. Validate, predict, filter by risk level, '
        'and download the results.</div>',
        unsafe_allow_html=True,
    )

    # ---- Step 1: choose the module ----
    _step(1, "Select ML Module")
    module_name = st.selectbox("Prediction module", MODULE_NAMES,
                               label_visibility="collapsed")
    config = get_module_config(module_name)
    required_columns = config["features"]

    # ---- Step 2: upload a file ----
    _step(2, "Upload Data File")
    required_columns_info(required_columns)
    sample_download_button(module_name, config["sample_file"])
    uploaded_file = upload_widget()

    if uploaded_file is None:
        st.caption("Tip: download the sample file above, fill it with your "
                   "data, and upload it here.")
        _render_results(module_name)
        return

    # Read the uploaded file into a DataFrame.
    try:
        input_df = read_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    # ---- Validation ----
    if is_empty(input_df):
        st.error("The uploaded file is empty. Please upload a file with data.")
        return

    is_valid, missing = validate_columns(input_df, required_columns)
    if not is_valid:
        st.error(
            "The file is missing required columns: "
            + ", ".join(f"`{c}`" for c in missing)
        )
        return

    st.success(f"File looks good — {input_df.shape[0]} rows, all required "
               f"columns present.")

    # ---- Step 3: preview ----
    _step(3, "Preview Uploaded Data")
    st.dataframe(input_df.head(5), use_container_width=True, hide_index=True)

    # ---- Step 4: run prediction ----
    _step(4, "Run Batch Prediction")
    if st.button("Run Batch Prediction", type="primary",
                 use_container_width=True):
        with st.spinner("Scoring all records..."):
            try:
                result_df = run_batch_prediction(module_name, input_df)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return
        # Persist so filters/downloads survive Streamlit reruns.
        st.session_state["batch_result"] = result_df
        st.session_state["batch_module"] = module_name

    # ---- Step 5: filter + download ----
    _render_results(module_name)


def _render_results(active_module):
    """Show KPIs, filters, table and download for the stored batch result."""
    result_df = st.session_state.get("batch_result")
    result_module = st.session_state.get("batch_module")

    if result_df is None:
        return

    _step(5, "Filter and Download Results")

    if result_module != active_module:
        st.warning(f"Showing earlier results for **{result_module}**. "
                   f"Upload a file and run again for **{active_module}**.")

    # KPI summary cards.
    summary_kpis(result_df, result_module)

    # ---- Filters ----
    st.markdown("**Filters**")
    f1, f2 = st.columns(2)
    with f1:
        categories = sorted(result_df["risk_category"].unique().tolist())
        selected_categories = st.multiselect("Risk Category", categories,
                                             default=categories)
    with f2:
        actions = sorted(result_df["business_action"].unique().tolist())
        selected_actions = st.multiselect("Business Action", actions,
                                          default=actions)

    p1, p2 = st.columns(2)
    with p1:
        min_prob = st.slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
    with p2:
        max_prob = st.slider("Maximum probability", 0.0, 1.0, 1.0, 0.05)

    # Apply filters.
    filtered = result_df[
        result_df["risk_category"].isin(selected_categories)
        & result_df["business_action"].isin(selected_actions)
        & (result_df["prediction_probability"] >= min_prob)
        & (result_df["prediction_probability"] <= max_prob)
    ]

    st.caption(f"Showing {len(filtered)} of {len(result_df)} records.")
    results_table(filtered)

    # ---- Download ----
    download_name = get_module_config(result_module)["download_name"]
    st.download_button(
        "Download Prediction Results",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
