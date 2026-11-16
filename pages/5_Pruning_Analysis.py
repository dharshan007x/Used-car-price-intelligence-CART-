"""
Page 5: Cost-Complexity Pruning Analysis
"""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import plot_pruning_curves

st.set_page_config(page_title="Pruning Analysis | Used Car Intelligence", page_icon="✂️", layout="wide")

st.markdown("# ✂️ Cost-Complexity Pruning (CCP) Analysis")
st.caption("Investigate Minimal Cost-Complexity Pruning: R_alpha(T) = R(T) + alpha * |T|")

reports_dir = PROJECT_ROOT / "reports"
models_dir = PROJECT_ROOT / "models"
pruning_file = reports_dir / "pruning_analysis.csv"
meta_file = models_dir / "model_metadata.json"

if pruning_file.exists() and meta_file.exists():
    pruning_df = pd.read_csv(pruning_file)
    with open(meta_file, "r") as f:
        meta = json.load(f)

    st.plotly_chart(plot_pruning_curves(pruning_df), use_container_width=True)

    st.markdown(f"**Pruning Summary**: {meta['pruning_interpretation']['summary']}")

    st.markdown("### Cost-Complexity Pruning Trajectory Table")
    st.dataframe(pruning_df, use_container_width=True)
else:
    st.warning("Pruning report not found. Please run: `python -m src.train`")
