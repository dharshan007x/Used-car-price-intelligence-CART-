"""
Page 1: Dataset Explorer & Validation Quality Audit
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_PROCESSED_PATH
from src.eda import compute_statistical_summary

st.set_page_config(page_title="Dataset Explorer | Used Car Intelligence", page_icon="📁", layout="wide")

st.markdown("# 📁 Dataset Explorer & Quality Audit")
st.caption("Inspect raw records, validation filters, schema properties, and exported cleaned data.")

if DEFAULT_PROCESSED_PATH.exists():
    df = pd.read_csv(DEFAULT_PROCESSED_PATH)

    tab1, tab2, tab3 = st.tabs(["🔍 Interactive Preview", "📊 Statistical Summary", "🛡️ Validation & Audit Report"])

    with tab1:
        st.markdown("### Cleaned Dataset Filter & Preview")
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_brand = st.selectbox("Filter Brand", ["All Brands"] + sorted(df["brand"].dropna().unique().tolist()))
        with c2:
            selected_fuel = st.selectbox("Filter Fuel", ["All Fuels"] + sorted(df["fuel"].dropna().unique().tolist()))
        with c3:
            selected_trans = st.selectbox("Filter Transmission", ["All Transmissions"] + sorted(df["transmission"].dropna().unique().tolist()))

        filtered_df = df.copy()
        if selected_brand != "All Brands":
            filtered_df = filtered_df[filtered_df["brand"] == selected_brand]
        if selected_fuel != "All Fuels":
            filtered_df = filtered_df[filtered_df["fuel"] == selected_fuel]
        if selected_trans != "All Transmissions":
            filtered_df = filtered_df[filtered_df["transmission"] == selected_trans]

        st.write(f"Showing **{len(filtered_df):,}** records matching filters:")
        st.dataframe(filtered_df.head(100), use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_used_car_data.csv",
            mime="text/csv"
        )

    with tab2:
        st.markdown("### 8-Point Numerical Summary")
        stat_df = compute_statistical_summary(df)
        st.dataframe(stat_df, use_container_width=True)

    with tab3:
        st.markdown("### Data Validation Rules & Safeguards")
        st.markdown("""
        - **Selling Price Validation**: Bounded between ₹25,000 and ₹5,00,00,000 to reject erroneous test records.
        - **Manufacturing Year**: Bounded between 1985 and 2024.
        - **Kilometers Driven**: Bounded between 0 and 10,00,000 km.
        - **Engine Displacement**: Validated between 600 CC and 6,500 CC.
        - **Train-Test Leakage Protection**: Imputation and One-Hot Encoding fitted strictly on the training partition.
        """)
else:
    st.warning("Processed dataset not found. Please run the training pipeline first: `python -m src.train`")
