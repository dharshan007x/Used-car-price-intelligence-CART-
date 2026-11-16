"""
Page 4: Tree Depth Analysis
"""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import plot_depth_experiment_curves

st.set_page_config(page_title="Depth Analysis | Used Car Intelligence", page_icon="🌲", layout="wide")

st.markdown("# 🌲 Tree Depth Experimentation & Bias-Variance Analysis")
st.caption("Empirical investigation of maximum tree depth constraints, underfitting, and overfitting in CART.")

reports_dir = PROJECT_ROOT / "reports"
models_dir = PROJECT_ROOT / "models"
depth_file = reports_dir / "depth_analysis.csv"
meta_file = models_dir / "model_metadata.json"

if depth_file.exists() and meta_file.exists():
    depth_df = pd.read_csv(depth_file)
    with open(meta_file, "r") as f:
        meta = json.load(f)

    st.plotly_chart(plot_depth_experiment_curves(depth_df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📉 Underfitting (High Bias)")
        st.write(meta["depth_interpretation"]["underfitting_diagnosis"])
    with c2:
        st.markdown("### 📈 Overfitting (High Variance)")
        st.write(meta["depth_interpretation"]["overfitting_diagnosis"])

    st.success(f"**Optimal Tradeoff**: {meta['depth_interpretation']['optimal_tradeoff_summary']}")

    st.markdown("### Experimental Results by Depth")
    st.dataframe(depth_df, use_container_width=True)
else:
    st.warning("Depth experiment results not found. Please run: `python -m src.train`")
