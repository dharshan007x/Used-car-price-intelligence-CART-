"""
Page 2: Exploratory Data Analysis — extended with ownership, seller type, segment tabs.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_PROCESSED_PATH
from src.eda import generate_dynamic_eda_insights
from src.visualization import (
    plot_price_distribution,
    plot_scatter_with_trend,
    plot_brand_analysis,
    plot_categorical_comparison,
    plot_correlation_heatmap,
)

st.set_page_config(
    page_title="EDA | Used Car Intelligence", page_icon="📈", layout="wide"
)

st.markdown("# 📈 Exploratory Data Analysis")
st.caption(
    "Deep-dive into vehicle attribute distributions, depreciation curves, ownership patterns, "
    "seller behaviour, and market segments."
)

if not DEFAULT_PROCESSED_PATH.exists():
    st.warning("Processed dataset not found. Please run: `python -m src.train`")
    st.stop()

df = pd.read_csv(DEFAULT_PROCESSED_PATH)

tabs = st.tabs([
    "📊 Price Distribution",
    "📉 Attribute Relationships",
    "🏷️ Brand & Fuel",
    "👤 Ownership & Seller",
    "🚗 Body Segment & Trim",
    "🔥 Correlation Heatmap",
    "💡 Dynamic Insights",
])

# ── Tab 1: Price Distribution ─────────────────────────────────────
with tabs[0]:
    st.plotly_chart(plot_price_distribution(df), use_container_width=True)

# ── Tab 2: Attribute Relationships ───────────────────────────────
with tabs[1]:
    c1, c2 = st.columns(2)
    num_opts = [c for c in ["vehicle_age", "km_driven", "engine_cc", "max_power_bhp",
                             "mileage_val", "torque_nm", "owner_tier", "seats"] if c in df.columns]
    cat_opts = [c for c in ["fuel", "transmission", "price_segment", "body_segment", "trim_tier"] if c in df.columns]
    with c1:
        attr = st.selectbox("Feature vs Price", num_opts)
    with c2:
        cat = st.selectbox("Color By", cat_opts)
    st.plotly_chart(plot_scatter_with_trend(df, x_col=attr, color_col=cat), use_container_width=True)

# ── Tab 3: Brand & Fuel ───────────────────────────────────────────
with tabs[2]:
    col_b1, col_b2 = st.columns(2)
    fig_c, fig_p = plot_brand_analysis(df, top_n=15)
    with col_b1:
        st.plotly_chart(fig_c, use_container_width=True)
    with col_b2:
        st.plotly_chart(fig_p, use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(plot_categorical_comparison(df, "fuel"), use_container_width=True)
    with col_c2:
        st.plotly_chart(plot_categorical_comparison(df, "transmission"), use_container_width=True)

# ── Tab 4: Ownership & Seller ─────────────────────────────────────
with tabs[3]:
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        if "owner" in df.columns and "selling_price" in df.columns:
            owner_order = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
            owner_order = [o for o in owner_order if o in df["owner"].unique()]
            fig_owner = px.box(
                df, x="owner", y="selling_price",
                category_orders={"owner": owner_order},
                title="Price Distribution by Owner Count",
                labels={"owner": "Ownership", "selling_price": "Selling Price (₹)"},
                color="owner",
            )
            fig_owner.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_owner, use_container_width=True)
        else:
            st.info("Owner column not available.")

    with col_o2:
        if "seller_type" in df.columns and "selling_price" in df.columns:
            fig_seller = px.box(
                df, x="seller_type", y="selling_price",
                title="Price Distribution by Seller Type",
                labels={"seller_type": "Seller Type", "selling_price": "Selling Price (₹)"},
                color="seller_type",
            )
            fig_seller.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_seller, use_container_width=True)
        else:
            st.info("seller_type column not available.")

    # Owner count vs mileage
    if "owner_tier" in df.columns and "km_driven" in df.columns:
        fig_ot = px.scatter(
            df, x="owner_tier", y="km_driven",
            color="selling_price",
            color_continuous_scale="Viridis",
            title="Owner Tier vs Kilometres Driven",
            labels={"owner_tier": "Owner Tier (1=First)", "km_driven": "km Driven"},
            opacity=0.5,
        )
        fig_ot.update_layout(template="plotly_white")
        st.plotly_chart(fig_ot, use_container_width=True)

    # Summary table
    if "owner" in df.columns:
        st.markdown("#### Median Price by Owner Count")
        owner_summary = df.groupby("owner")["selling_price"].agg(
            Count="count", Median="median", Mean="mean"
        ).reset_index().sort_values("Median", ascending=False)
        owner_summary["Median"] = owner_summary["Median"].map(lambda x: f"₹{x:,.0f}")
        owner_summary["Mean"] = owner_summary["Mean"].map(lambda x: f"₹{x:,.0f}")
        st.dataframe(owner_summary, use_container_width=True)

# ── Tab 5: Body Segment & Trim ────────────────────────────────────
with tabs[4]:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if "body_segment" in df.columns and "selling_price" in df.columns:
            fig_seg = px.box(
                df, x="body_segment", y="selling_price",
                title="Price by Body Segment",
                labels={"body_segment": "Body Segment", "selling_price": "Selling Price (₹)"},
                color="body_segment",
            )
            fig_seg.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.info("body_segment column not available. Run training first.")

    with col_s2:
        if "trim_tier" in df.columns and "selling_price" in df.columns:
            fig_trim = px.box(
                df, x="trim_tier", y="selling_price",
                category_orders={"trim_tier": ["Base", "Mid", "Top"]},
                title="Price by Trim Tier",
                labels={"trim_tier": "Trim Tier", "selling_price": "Selling Price (₹)"},
                color="trim_tier",
                color_discrete_map={"Base": "#9ecae1", "Mid": "#4292c6", "Top": "#08519c"},
            )
            fig_trim.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_trim, use_container_width=True)
        else:
            st.info("trim_tier column not available. Run training first.")

    # Segment count bar
    if "body_segment" in df.columns:
        seg_counts = df["body_segment"].value_counts().reset_index()
        seg_counts.columns = ["body_segment", "count"]
        fig_seg_cnt = px.bar(
            seg_counts, x="body_segment", y="count",
            title="Listing Count by Body Segment",
            color="body_segment",
        )
        fig_seg_cnt.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_seg_cnt, use_container_width=True)

# ── Tab 6: Correlation Heatmap ────────────────────────────────────
with tabs[5]:
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

# ── Tab 7: Dynamic Insights ───────────────────────────────────────
with tabs[6]:
    st.markdown("### 💡 Data-Grounded Market Insights")
    insights = generate_dynamic_eda_insights(df)
    for item in insights:
        st.markdown(f"**📌 {item['category']}**: {item['finding']}")
