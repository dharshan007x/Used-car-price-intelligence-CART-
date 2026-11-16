"""
Page 7: Decision Tree Structure Explorer
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_pipeline
from src.visualization import render_decision_tree_matplotlib

st.set_page_config(page_title="Tree Explorer | Used Car Intelligence", page_icon="🌳", layout="wide")

st.markdown("# 🌳 Decision Tree Architecture Explorer")
st.caption("Inspect learned CART node splits, thresholds, and leaf price values.")

models_dir = PROJECT_ROOT / "models"
meta_file = models_dir / "model_metadata.json"
model_file = models_dir / "best_cart_model.joblib"

if meta_file.exists() and model_file.exists():
    with open(meta_file, "r") as f:
        meta = json.load(f)
    model = load_pipeline(model_file)

    depth_display = st.slider("Select Display Depth", min_value=1, max_value=4, value=2)
    st.info(f"Displaying top {depth_display} levels of the CART model ({meta['tree_properties']['total_nodes']:,} total nodes).")

    feat_names = meta["transformed_feature_names"]
    fig = render_decision_tree_matplotlib(model, feat_names, max_depth_to_display=depth_display)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Model artifacts not found. Please train models first: `python -m src.train`")
