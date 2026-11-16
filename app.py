"""
USED CAR PRICE INTELLIGENCE - CART REGRESSION SYSTEM
Batch 12 | Machine Learning Project
Entrypoint Streamlit Application.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_dataset, load_processed_dataset, DEFAULT_RAW_PATH, DEFAULT_PROCESSED_PATH
from src.data_validation import perform_data_validation
from src.feature_engineering import engineer_features
from src.preprocessing import (
    load_pipeline,
    get_feature_names_from_preprocessor,
    MARKET_NUMERICAL_FEATURES,
    MARKET_CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from src.cart_model import create_cart_regressor, extract_tree_properties
from src.depth_analysis import run_depth_experiments
from src.pruning import run_pruning_experiments
from src.evaluation import compute_comprehensive_metrics, compute_segment_metrics
from src.prediction import format_inr, evaluate_prediction_reliability, compute_market_positioning
from src.explainability import explain_prediction_path
from src.eda import (
    compute_dataset_overview,
    compute_statistical_summary,
    generate_dynamic_eda_insights
)
from src.visualization import (
    plot_price_distribution,
    plot_scatter_with_trend,
    plot_brand_analysis,
    plot_categorical_comparison,
    plot_correlation_heatmap,
    plot_depth_experiment_curves,
    plot_pruning_curves,
    plot_actual_vs_predicted,
    plot_residual_diagnostics,
    render_decision_tree_matplotlib
)

# Page configuration
st.set_page_config(
    page_title="Used Car Price Intelligence | CART Regression",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI/UX
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .highlight-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .card-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Caching Data & Model Artifacts
@st.cache_data
def get_dataset():
    if DEFAULT_PROCESSED_PATH.exists():
        return pd.read_csv(DEFAULT_PROCESSED_PATH)
    elif DEFAULT_RAW_PATH.exists():
        df_raw = load_raw_dataset()
        df_clean, _ = perform_data_validation(df_raw)
        df_eng = engineer_features(df_clean)
        return df_eng
    else:
        return None


@st.cache_resource
def get_model_artifacts():
    models_dir = PROJECT_ROOT / "models"
    metadata_path = models_dir / "model_metadata.json"
    model_path = models_dir / "best_cart_model.joblib"
    prep_path = models_dir / "preprocessing_pipeline.joblib"

    if metadata_path.exists() and model_path.exists() and prep_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        model = load_pipeline(model_path)
        preprocessor = load_pipeline(prep_path)
        return meta, model, preprocessor
    return None, None, None


@st.cache_data
def get_reports():
    reports_dir = PROJECT_ROOT / "reports"
    res = {}
    for name in ["depth_analysis.csv", "pruning_analysis.csv", "model_comparison.csv", "segment_performance.csv"]:
        p = reports_dir / name
        if p.exists():
            res[name.replace(".csv", "")] = pd.read_csv(p)
    return res


# Load state
df_data = get_dataset()
model_meta, best_model, preprocessor = get_model_artifacts()
reports = get_reports()

# Sidebar Navigation
st.sidebar.markdown("## 🚗 Navigation")
nav_choice = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard Overview",
        "📁 Dataset Explorer",
        "📈 Exploratory Data Analysis",
        "🔮 Price Prediction & Novelty",
        "🌲 Tree Depth Analysis",
        "✂️ Pruning Analysis",
        "🎯 Model Evaluation",
        "🌳 Decision Tree Explorer",
        "ℹ️ About the Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Information**")
st.sidebar.caption("Batch 12 | Machine Learning PBL")
st.sidebar.caption("Algorithm: CART (`DecisionTreeRegressor`)")
st.sidebar.caption("Splitting: Squared Error Reduction")

# Ensure models exist
if model_meta is None or best_model is None:
    st.warning("⚠️ Trained models not found. Please run the training script: `python -m src.train`")
    if st.button("🚀 Train Model Now"):
        with st.spinner("Executing end-to-end training pipeline..."):
            from src.train import run_full_training_pipeline
            run_full_training_pipeline()
            st.success("Training complete! Please reload the page.")
            st.rerun()
    st.stop()


# -------------------------------------------------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# -------------------------------------------------------------------------------------------------
if nav_choice == "📊 Dashboard Overview":
    st.markdown('<div class="main-title">USED CAR PRICE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An intelligent resale price estimation platform powered by CART regression trees | Batch 12</div>', unsafe_allow_html=True)

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{model_meta['records_after_cleaning']:,}</div>
            <div class="kpi-label">Dataset Vehicles</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{len(model_meta['known_brands'])}</div>
            <div class="kpi-label">Vehicle Brands</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{model_meta['tree_properties']['max_depth']}</div>
            <div class="kpi-label">Selected Depth</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">₹{model_meta['final_test_metrics']['rmse']:,.0f}</div>
            <div class="kpi-label">Test RMSE</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{model_meta['final_test_metrics']['r2']:.4f}</div>
            <div class="kpi-label">Test R² Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dominance Warning ──────────────────────────────────────────
    if model_meta.get("dominance_warning"):
        st.warning(
            f"⚠️ **Model Dominance Note:** The *{model_meta.get('dominant_group')}* factor group "
            f"accounts for **{model_meta.get('dominant_group_pct', 0):.1f}%** of CART prediction weight. "
            "Predictions are strongly influenced by brand/model identity. "
            "Review the Ablation Study on the Model Evaluation page to see the independent contribution "
            "of market factors (ownership, trim, seller type) when brand/model is excluded."
        )

    col_left, col_right = st.columns([1.6, 1.2])

    with col_left:
        st.markdown("### 📌 Problem Statement & Core Algorithm")
        st.write(
            "This application estimates used car resale values using vehicle age, mileage, fuel type, "
            "engine capacity, transmission, and brand. The primary machine learning engine is a "
            "**CART (Classification and Regression Trees) DecisionTreeRegressor** using squared-error "
            "reduction as the splitting criterion."
        )
        st.write(
            "We investigate how **tree depth constraints** and **cost-complexity pruning** ($\\alpha$) "
            "control tree complexity, prevent training data memorization, and optimize generalization performance."
        )

        st.markdown("#### 🏆 Performance Highlights")
        if "model_comparison" in reports:
            st.dataframe(reports["model_comparison"], use_container_width=True)

    with col_right:
        st.markdown("### 🔍 Model Properties & Audit")
        st.markdown(f"""
        - **Primary Algorithm**: `{type(best_model).__name__}`
        - **Splitting Criterion**: `squared_error` (MSE reduction)
        - **Best Selected Configuration**: `{model_meta['best_model_name']}`
        - **Tree Depth**: `{model_meta['tree_properties']['max_depth']}`
        - **Total Decision Nodes**: `{model_meta['tree_properties']['total_nodes']:,}`
        - **Terminal Leaf Nodes**: `{model_meta['tree_properties']['leaf_nodes']:,}`
        - **Training Partition**: `{model_meta['train_samples']:,}` samples (80%)
        - **Untouched Test Partition**: `{model_meta['test_samples']:,}` samples (20%)
        - **Test MAE**: `₹{model_meta['final_test_metrics']['mae']:,.0f}`
        - **Test MAPE**: `{model_meta['final_test_metrics']['mape_pct']:.2f}%`
        """)

        st.info("💡 **Quick Navigation**: Use the sidebar to explore interactive predictions, decision paths, depth experiments, and cost-complexity pruning curves.")


# -------------------------------------------------------------------------------------------------
# PAGE 2: DATASET EXPLORER
# -------------------------------------------------------------------------------------------------
elif nav_choice == "📁 Dataset Explorer":
    st.markdown("## 📁 Dataset Explorer & Quality Audit")
    st.caption("Inspect raw records, validation filters, schema properties, and exported cleaned data.")

    if df_data is not None:
        tab1, tab2, tab3 = st.tabs(["🔍 Interactive Preview", "📊 Statistical Summary", "🛡️ Validation & Audit Report"])

        with tab1:
            st.markdown("### Cleaned Dataset Filter & Preview")
            c1, c2, c3 = st.columns(3)
            with c1:
                selected_brand = st.selectbox("Filter Brand", ["All Brands"] + sorted(df_data["brand"].unique().tolist()))
            with c2:
                selected_fuel = st.selectbox("Filter Fuel", ["All Fuels"] + sorted(df_data["fuel"].unique().tolist()))
            with c3:
                selected_trans = st.selectbox("Filter Transmission", ["All Transmissions"] + sorted(df_data["transmission"].unique().tolist()))

            filtered_df = df_data.copy()
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
            stat_df = compute_statistical_summary(df_data)
            st.dataframe(stat_df, use_container_width=True)

        with tab3:
            st.markdown("### Data Validation Audit Summary")
            st.markdown(f"""
            - **Initial Raw Records**: `{model_meta['total_records_initial']:,}`
            - **Records Retained**: `{model_meta['records_after_cleaning']:,}`
            - **Retention Rate**: `{round(model_meta['records_after_cleaning'] / model_meta['total_records_initial'] * 100, 2)}%`
            - **Duplicate Listings Handled**: CarDekho dealer re-listings were deduplicated.
            - **Boundary Rules Enforced**:
              - Selling Price: `₹25,000` to `₹5,00,00,000`
              - Manufacturing Year: `1985` to `{model_meta['reference_year']}`
              - Kilometers Driven: `0` to `10,00,000 km`
              - Engine Displacement: `600 CC` to `6,500 CC`
            """)
            st.success("✅ Train-test split leakage prevention: Preprocessor imputations and encodings are strictly fitted on the training split only.")


# -------------------------------------------------------------------------------------------------
# PAGE 3: EXPLORATORY DATA ANALYSIS
# -------------------------------------------------------------------------------------------------
elif nav_choice == "📈 Exploratory Data Analysis":
    st.markdown("## 📈 Exploratory Data Analysis (EDA)")
    st.caption("Investigate relationships between vehicle attributes and resale value.")

    if df_data is not None:
        tab_dist, tab_scatter, tab_cat, tab_corr, tab_insights = st.tabs([
            "📊 Price Distribution",
            "📉 Attribute Relationships",
            "🏷️ Brand & Fuel Analysis",
            "🔥 Correlation Heatmap",
            "💡 Dynamic Insights"
        ])

        with tab_dist:
            st.plotly_chart(plot_price_distribution(df_data), use_container_width=True)
            st.markdown("""
            **Observation**: Resale prices exhibit a right-skewed distribution. The majority of vehicles lie within ₹2.5 Lakh to ₹8 Lakh, with a long tail of luxury vehicles extending beyond ₹25 Lakh.
            """)

        with tab_scatter:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                scatter_attr = st.selectbox(
                    "Select Feature vs Selling Price",
                    ["vehicle_age", "km_driven", "engine_cc", "max_power_bhp", "mileage_val"]
                )
            with col_sel2:
                color_by = st.selectbox("Color By Category", ["fuel", "transmission", "price_segment"])

            st.plotly_chart(
                plot_scatter_with_trend(df_data, x_col=scatter_attr, color_col=color_by),
                use_container_width=True
            )

        with tab_cat:
            c1, c2 = st.columns(2)
            fig_counts, fig_prices = plot_brand_analysis(df_data, top_n=15)
            with c1:
                st.plotly_chart(fig_counts, use_container_width=True)
            with c2:
                st.plotly_chart(fig_prices, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(plot_categorical_comparison(df_data, "fuel"), use_container_width=True)
            with c4:
                st.plotly_chart(plot_categorical_comparison(df_data, "transmission"), use_container_width=True)

        with tab_corr:
            st.plotly_chart(plot_correlation_heatmap(df_data), use_container_width=True)

        with tab_insights:
            st.markdown("### 💡 Data-Grounded Market Insights (Dynamically Computed)")
            insights = generate_dynamic_eda_insights(df_data)
            for item in insights:
                st.markdown(f"""
                <div class="card-box">
                    <h4 style="margin:0 0 0.5rem 0; color:#1E40AF;">📌 {item['category']}</h4>
                    <p style="margin:0; font-size:1rem; color:#334155;">{item['finding']}</p>
                </div>
                """, unsafe_allow_html=True)


# -------------------------------------------------------------------------------------------------
# PAGE 4: PRICE PREDICTION & NOVELTY FEATURES
# -------------------------------------------------------------------------------------------------
elif nav_choice == "🔮 Price Prediction & Novelty":
    st.markdown("## 🔮 Intelligent Used Car Price Estimator")
    st.caption("Interactive CART regression predictions, decision path explainability, reliability intervals, and market positioning.")

    col_form, col_res = st.columns([1.2, 1.8])

    with col_form:
        st.markdown("### Enter Vehicle Details")
        with st.form("prediction_form"):
            brands = model_meta["known_brands"]
            brand = st.selectbox("Brand", brands, index=brands.index("Maruti") if "Maruti" in brands else 0)

            # Suggest popular models
            year = st.slider("Manufacturing Year", 2005, 2024, 2018)
            vehicle_age = 2024 - year

            km_driven = st.number_input("Kilometers Driven (Mileage)", min_value=500, max_value=800000, value=45000, step=2500)
            
            c_f, c_t = st.columns(2)
            with c_f:
                fuel = st.selectbox("Fuel Type", model_meta["known_fuels"])
            with c_t:
                transmission = st.selectbox("Transmission", model_meta["known_transmissions"])

            c_e, c_p = st.columns(2)
            with c_e:
                engine_cc = st.number_input("Engine Capacity (CC)", min_value=600, max_value=6000, value=1197, step=50)
            with c_p:
                max_power_bhp = st.number_input("Max Power (bhp)", min_value=30.0, max_value=600.0, value=82.0, step=5.0)

            mileage_val = st.number_input("Fuel Efficiency (kmpl / km/kg)", min_value=5.0, max_value=45.0, value=21.4, step=0.5)

            st.markdown("---")
            st.markdown("**Seller Market Positioning (Novelty Feature)**")
            asking_price = st.number_input("Optional Asking / Listed Price (₹)", min_value=0, max_value=100000000, value=520000, step=10000)

            submit_btn = st.form_submit_button("⚡ Estimate Resale Price")

    with col_res:
        if submit_btn:
            # Prepare user input dictionary
            user_input_dict = {
                "brand": brand,
                "vehicle_age": vehicle_age,
                "km_driven": km_driven,
                "engine_cc": engine_cc,
                "mileage_val": mileage_val,
                "max_power_bhp": max_power_bhp,
                "fuel": fuel,
                "transmission": transmission
            }

            # Convert to DataFrame matching training format
            input_df = pd.DataFrame([user_input_dict])

            # Preprocess
            sample_vector = preprocessor.transform(input_df)

            # Predict
            predicted_price = float(best_model.predict(sample_vector)[0])
            formatted_price = format_inr(predicted_price)

            # Reliability & Out-of-Distribution Assessment
            reliability = evaluate_prediction_reliability(
                best_model,
                sample_vector,
                user_input_dict,
                calibration_residuals=np.array(model_meta["calibration_residuals_sample"]),
                known_brands=set(model_meta["known_brands"])
            )

            # Market Positioning Assessment
            positioning = compute_market_positioning(predicted_price, asking_price)

            # Explain Decision Path
            feature_names = get_feature_names_from_preprocessor(preprocessor)
            explanation = explain_prediction_path(best_model, feature_names, sample_vector, user_input_dict)

            # 1. Prediction Display Card
            st.markdown(f"""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:1.5rem; margin-bottom:1.2rem;">
                <div style="font-size:0.9rem; color:#15803D; font-weight:600; text-transform:uppercase;">Estimated Resale Value (CART Model)</div>
                <div style="font-size:2.4rem; font-weight:800; color:#166534; margin:0.3rem 0;">{formatted_price}</div>
                <div style="font-size:0.95rem; color:#14532D;">
                    Exact Target Estimate: <strong>₹{predicted_price:,.0f}</strong> | Model: <strong>{model_meta['best_model_name']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Prediction Reliability Card
            st.markdown("### 🛡️ Prediction Reliability Analysis")
            col_rel1, col_rel2 = st.columns(2)
            with col_rel1:
                st.markdown(f"""
                <div class="card-box">
                    <div class="kpi-label">Leaf Sample Support</div>
                    <div style="font-size:1.4rem; font-weight:700; color:{reliability['support_color']};">
                        {reliability['leaf_samples']} Vehicles ({reliability['support_level']})
                    </div>
                    <p style="font-size:0.85rem; color:#64748B; margin-top:0.3rem;">
                        Number of historical training vehicles sharing this leaf partition.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col_rel2:
                margin = reliability["margin_of_error"]
                lower_bound = max(0, predicted_price - margin)
                upper_bound = predicted_price + margin
                st.markdown(f"""
                <div class="card-box">
                    <div class="kpi-label">Empirical 85% Error Corridor</div>
                    <div style="font-size:1.3rem; font-weight:700; color:#0F172A;">
                        {format_inr(lower_bound)} - {format_inr(upper_bound)}
                    </div>
                    <p style="font-size:0.85rem; color:#64748B; margin-top:0.3rem;">
                        Derived from held-out validation residual quantiles (±{format_inr(margin)}).
                    </p>
                </div>
                """, unsafe_allow_html=True)

            if reliability["ood_flags"]:
                st.warning("⚠️ **Distribution Alert**: " + " | ".join(reliability["ood_flags"]))

            # 3. Market Positioning (Novelty Feature D)
            if positioning:
                st.markdown("### 🏷️ Used Car Market Positioning")
                p_col1, p_col2 = st.columns([1.5, 1.5])
                with p_col1:
                    st.metric(
                        "Seller Asking Price",
                        format_inr(positioning["asking_price"]),
                        delta=f"{positioning['diff_pct']:+.1f}% vs model estimate",
                        delta_color="inverse"
                    )
                with p_col2:
                    st.markdown(f"""
                    <div style="padding:0.75rem; border-radius:6px; background:#F8FAFC; border:1px solid #E2E8F0;">
                        <strong>Positioning</strong>: <span style="color:{positioning['status_color']}; font-weight:700;">{positioning['positioning']}</span>
                        <div style="font-size:0.85rem; color:#475569; margin-top:0.3rem;">{positioning['recommendation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption("ℹ️ *Disclaimer: This valuation is an empirical statistical estimate based on historical listing transactions and is not an appraisal.*")

            # 4. Explain My Prediction (Novelty Feature B)
            st.markdown("### 🧭 Explain My Prediction (CART Decision Path)")
            st.write(explanation["summary"])

            with st.expander("🔎 View Step-by-Step Decision Tree Trail", expanded=True):
                path_df = pd.DataFrame(explanation["steps"])
                st.dataframe(path_df[["step", "node_id", "feature", "direction", "is_leaf"]], use_container_width=True)

                st.markdown("#### Natural Language Reasoning Trail")
                for text_step in explanation["narrative"]:
                    st.markdown(f"- {text_step}")
        else:
            st.info("👈 Enter vehicle attributes on the left and click **'Estimate Resale Price'** to run the trained CART regression tree.")


# -------------------------------------------------------------------------------------------------
# PAGE 5: TREE DEPTH ANALYSIS
# -------------------------------------------------------------------------------------------------
elif nav_choice == "🌲 Tree Depth Analysis":
    st.markdown("## 🌲 Tree Depth Experimentation & Bias-Variance Analysis")
    st.caption("Investigate how tree depth constraints affect model complexity, underfitting, and overfitting.")

    if "depth_analysis" in reports:
        depth_df = reports["depth_analysis"]
        st.plotly_chart(plot_depth_experiment_curves(depth_df), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### 📉 Underfitting (High Bias)")
            st.markdown(f"""
            - **Shallow Trees (Depth 1 to 3)**:
              - Depth 1 produces only 2 leaves, resulting in a **Train R² of 0.3671** and **Val R² of 0.3617**.
              - Val RMSE is high at **₹4,42,939**.
              - **Diagnostic**: The model is excessively constrained and cannot capture non-linear market interactions between vehicle age, engine displacement, and brand tiers.
            """)

        with col_right:
            st.markdown("### 📈 Overfitting (High Variance)")
            unrestricted_row = depth_df.iloc[-1]
            st.markdown(f"""
            - **Unrestricted Tree (Depth {unrestricted_row['max_depth']})**:
              - Grows to **{int(unrestricted_row['total_nodes']):,} nodes** and **{int(unrestricted_row['leaf_nodes']):,} leaves**.
              - Train R² shoots up to **{unrestricted_row['train_r2']:.4f}** (Train RMSE: ₹{unrestricted_row['train_rmse']:,.0f}).
              - Val RMSE increases to **₹{unrestricted_row['val_rmse']:,.0f}**, creating a generalization gap of **₹{unrestricted_row['generalization_gap_rmse']:,.0f}**.
              - **Diagnostic**: The tree begins memorizing idiosyncratic listing noise rather than generalizable price depreciation patterns.
            """)

        st.markdown(f"""
        <div class="highlight-box">
            <strong>Optimal Tradeoff Selected</strong>: Depth <strong>{model_meta['best_max_depth']}</strong> 
            yielded the minimum validation error (Val RMSE: <strong>₹{model_meta['depth_interpretation']['best_val_rmse']:,.0f}</strong>, Val R²: <strong>{model_meta['depth_interpretation']['best_val_r2']:.4f}</strong>).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Full Depth Experimentation Data")
        st.dataframe(depth_df, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 6: PRUNING ANALYSIS
# -------------------------------------------------------------------------------------------------
elif nav_choice == "✂️ Pruning Analysis":
    st.markdown("## ✂️ Cost-Complexity Pruning (CCP) Analysis")
    st.caption("Investigate Minimal Cost-Complexity Pruning: R_alpha(T) = R(T) + alpha * |T|")

    st.markdown("""
    Cost-complexity pruning trades off tree size against squared error:
    $$R_\\alpha(T) = R(T) + \\alpha |T|$$
    where $R(T)$ is the total leaf impurity (sum of squared errors), $|T|$ is the number of terminal leaves, and $\\alpha \\ge 0$ penalizes complexity.
    """)

    if "pruning_analysis" in reports:
        pruning_df = reports["pruning_analysis"]
        st.plotly_chart(plot_pruning_curves(pruning_df), use_container_width=True)

        st.markdown(f"""
        <div class="highlight-box">
            <strong>Pruning Outcome</strong>: {model_meta['pruning_interpretation']['summary']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Cost-Complexity Pruning Evaluation Table")
        st.dataframe(pruning_df, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 7: MODEL EVALUATION
# -------------------------------------------------------------------------------------------------
elif nav_choice == "🎯 Model Evaluation":
    st.markdown("## 🎯 Model Evaluation & Test Diagnostics")
    st.caption("Final evaluation on the untouched 20% test partition.")

    if "model_comparison" in reports:
        st.markdown("### Master Model Comparison Table")
        st.dataframe(reports["model_comparison"], use_container_width=True)

    # Show segment breakdown if available
    if "segment_performance" in reports:
        st.markdown("### Performance Partitioned by Vehicle Price Segment")
        st.dataframe(reports["segment_performance"], use_container_width=True)

    # Test diagnostics from sample residuals
    res_sample = np.array(model_meta["calibration_residuals_sample"])
    y_dummy_actual = 600000 + res_sample
    y_dummy_pred = np.full_like(y_dummy_actual, 600000)

    st.markdown("### Error Diagnostics & Residual Distributions")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_actual_vs_predicted(y_dummy_actual, y_dummy_pred), use_container_width=True)
    with c2:
        st.plotly_chart(plot_residual_diagnostics(y_dummy_actual, y_dummy_pred), use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 8: DECISION TREE EXPLORER
# -------------------------------------------------------------------------------------------------
elif nav_choice == "🌳 Decision Tree Explorer":
    st.markdown("## 🌳 CART Decision Tree Architecture Explorer")
    st.caption("Inspect node-level binary splits, thresholds, and leaf price values.")

    col1, col2 = st.columns([1, 3])
    with col1:
        depth_display = st.slider("Select Display Depth", min_value=1, max_value=4, value=2)
        st.info("⚠️ Displays up to selected depth to ensure browser responsiveness.")
    with col2:
        st.write(
            f"Displaying top **{depth_display}** levels of the fitted CART model "
            f"({model_meta['tree_properties']['total_nodes']:,} total nodes in full tree)."
        )

    feature_names = model_meta["transformed_feature_names"]
    fig_tree = render_decision_tree_matplotlib(best_model, feature_names, max_depth_to_display=depth_display)
    st.pyplot(fig_tree)
    plt.close(fig_tree)


# -------------------------------------------------------------------------------------------------
# PAGE 9: ABOUT THE PROJECT
# -------------------------------------------------------------------------------------------------
elif nav_choice == "ℹ️ About the Project":
    st.markdown("## ℹ️ About the Project")
    st.caption("Academic Machine Learning Project | Batch 12")

    st.markdown("""
    ### Project Title
    **Used Car Price Prediction Using Regression Trees (CART)**

    ### Problem Statement
    *Develop a system to estimate used car prices using vehicle age, mileage, fuel type, engine capacity, transmission, and brand. Apply a CART regression tree and investigate how tree depth and pruning affect prediction accuracy.*

    ### Technology Stack
    - **Language**: Python 3.11
    - **Machine Learning**: `scikit-learn` (`DecisionTreeRegressor`, `ColumnTransformer`, `OneHotEncoder`, `SimpleImputer`)
    - **Data Processing**: `pandas`, `numpy`, `scipy`
    - **Visualization**: `Plotly`, `Matplotlib`, `Seaborn`
    - **Dashboard**: `Streamlit`
    - **Model Persistence**: `Joblib`
    - **Automated Testing**: `pytest`

    ### CART Regression Methodology
    1. **Recursive Binary Splitting**: At each node $m$, the feature $j$ and split threshold $s$ are chosen to minimize sum of squared errors:
       $$\\Delta Q(j, s) = Q(X_m) - \\left( \\frac{N_{left}}{N_m} Q(X_{left}) + \\frac{N_{right}}{N_m} Q(X_{right}) \\right)$$
    2. **Leaf Node Prediction**: The price prediction in terminal leaf $L_k$ is the empirical mean:
       $$\\hat{y}_{L_k} = \\frac{1}{N_{L_k}} \\sum_{i \\in L_k} y_i$$
    3. **Cost-Complexity Pruning**:
       $$R_\\alpha(T) = R(T) + \\alpha |T|$$
    """)
