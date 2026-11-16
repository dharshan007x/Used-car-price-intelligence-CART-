"""
Page 3: Market-Grade Price Prediction & Intelligence
Full multi-factor prediction with:
  - Brand → Model dropdown (complete, brand-specific, from full dataset)
  - Auto-suggestion of fuel/transmission/engine from dataset
  - 8 market-style outputs (range, reliability, comparables, what-if,
    depreciation, sensitivity, decision path, market positioning)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_pipeline, get_feature_names_from_preprocessor
from src.prediction import (
    format_inr,
    evaluate_prediction_reliability,
    compute_market_positioning,
    compute_price_range,
    get_comparable_listings,
    compute_depreciation_curve,
    compute_factor_sensitivity,
    compute_what_if,
    apply_mileage_adjustment,
    compute_mileage_impact_curve,
    _build_input_df,
)
from src.explainability import explain_prediction_path
from src.feature_engineering import encode_owner_tier

st.set_page_config(
    page_title="Price Prediction | Used Car Intelligence",
    page_icon="🔮",
    layout="wide",
)

st.markdown("# 🔮 Market-Grade Used Car Price Estimator")
st.caption(
    "Multi-factor CART — age, km, ownership, seller type, trim, body segment, power, torque. "
    "**Location/city is not in this dataset.**"
)

# ════════════════════════════════════════════════════════════════════
# Load Artifacts
# ════════════════════════════════════════════════════════════════════
models_dir = PROJECT_ROOT / "models"
meta_path = models_dir / "model_metadata.json"
model_path = models_dir / "best_cart_model.joblib"
prep_path = models_dir / "preprocessing_pipeline.joblib"
comp_path = models_dir / "comparable_records.csv"
map_path = models_dir / "brand_model_map.json"
catalog_path = PROJECT_ROOT / "data" / "reference" / "model_catalog.json"
adjuster_path = models_dir / "mileage_adjuster.joblib"

if not (meta_path.exists() and model_path.exists() and prep_path.exists()):
    st.warning("Model files not found. Please run: `python -m src.train`")
    st.stop()

with open(meta_path, "r") as f:
    meta = json.load(f)

best_model = load_pipeline(model_path)
preprocessor = load_pipeline(prep_path)
feat_names = get_feature_names_from_preprocessor(preprocessor)
all_feature_cols = meta.get("input_features", feat_names)
num_cols = meta.get("numerical_features", [])

comparable_df = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()

# ── Mileage adjuster (optional — may not exist in older runs) ───
mileage_adjuster = None
if adjuster_path.exists():
    try:
        mileage_adjuster = load_pipeline(adjuster_path)
    except Exception:
        mileage_adjuster = None

typical_km_per_year = float(meta.get("typical_km_per_year", 10000.0))

# ── Brand-Model Map ─────────────────────────────────────────────
brand_model_map: dict = {}
if map_path.exists():
    with open(map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    brand_model_map = raw_map.get("brands", {})

# ── Reference Catalog (optional catalog-only models) ───────────
SHOW_CATALOG_ONLY_MODELS = True
catalog_models: dict = {}
if catalog_path.exists() and SHOW_CATALOG_ONLY_MODELS:
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_raw = json.load(f)
    # Remove metadata keys
    catalog_models = {k: v for k, v in catalog_raw.items() if not k.startswith("_")}

known_brands = sorted(brand_model_map.keys())
known_fuels = meta.get("known_fuels", ["Petrol", "Diesel", "Cng", "Lpg"])
known_transmissions = meta.get("known_transmissions", ["Manual", "Automatic"])
known_seller_types = meta.get("known_seller_types", ["Individual", "Dealer", "Trustmark Dealer"])
known_owners = meta.get("known_owners", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
training_stats = meta.get("training_stats", {})
calibration_residuals = np.array(meta.get("calibration_residuals_sample", []))
residuals_by_band = {k: np.array(v) for k, v in meta.get("calibration_residuals_by_band", {}).items()}

# ── Auto-suggest defaults from comparable listings ──────────────
def _get_model_defaults(brand: str, model: str, comp_df: pd.DataFrame) -> dict:
    """Returns mode values for fuel, transmission, engine_cc, seats for brand/model."""
    defaults = {
        "fuel": known_fuels[0],
        "transmission": "Manual",
        "engine_cc": 1197,
        "max_power_bhp": 82.0,
        "mileage_val": 18.0,
        "torque_nm": 113.0,
        "seats": 5,
    }
    if comp_df.empty:
        return defaults
    subset = comp_df[(comp_df.get("brand", pd.Series()) == brand) &
                     (comp_df.get("model_name", pd.Series()) == model)] if \
        ("brand" in comp_df.columns and "model_name" in comp_df.columns) else pd.DataFrame()
    if subset.empty:
        subset = comp_df[comp_df.get("brand", pd.Series()) == brand] if \
            "brand" in comp_df.columns else pd.DataFrame()
    if subset.empty:
        return defaults
    for col, key in [("fuel", "fuel"), ("transmission", "transmission")]:
        if col in subset.columns:
            mode_val = subset[col].mode()
            if not mode_val.empty:
                defaults[key] = mode_val.iloc[0]
    return defaults


# ── Dominance warning ─────────────────────────────────────────
if meta.get("dominance_warning"):
    st.warning(
        f"⚠️ **Model Dominance Note:** *{meta.get('dominant_group')}* accounts for "
        f"**{meta.get('dominant_group_pct', 0):.1f}%** of prediction weight. "
        "Verify comparable listings below."
    )

# ════════════════════════════════════════════════════════════════════
# INPUT FORM
# ════════════════════════════════════════════════════════════════════
col_form, col_res = st.columns([1.2, 1.8])

with col_form:
    st.markdown("### 🚗 Vehicle Attributes")

    # ── Brand selection (outside form so model list updates immediately) ──
    brand = st.selectbox(
        "Brand",
        known_brands,
        index=known_brands.index("Maruti") if "Maruti" in known_brands else 0,
        key="brand_select",
    )

    # ── Build model list for selected brand ────────────────────────
    dataset_models = brand_model_map.get(brand, [])  # list of {name, count}
    dataset_model_names = [f"{m['name']} ({m['count']:,} listings)" for m in dataset_models]
    dataset_model_raw_names = [m["name"] for m in dataset_models]

    # Merge catalog-only models if enabled
    catalog_only_names = []
    if SHOW_CATALOG_ONLY_MODELS and brand in catalog_models:
        for cm in catalog_models[brand]:
            if cm not in dataset_model_raw_names:
                catalog_only_names.append(cm)

    # Combined list: dataset models first (with count), then catalog-only, then Other
    all_model_display = dataset_model_names.copy()
    all_model_raw = dataset_model_raw_names.copy()
    for cm in catalog_only_names:
        all_model_display.append(f"{cm} (no listings in dataset)")
        all_model_raw.append(cm)
    all_model_display.append("Other / not listed")
    all_model_raw.append("Other")

    st.caption(f"**{len(dataset_model_raw_names)}** models with dataset listings, "
               f"**{len(catalog_only_names)}** catalog-only.")

    # Reset model selection when brand changes
    if "prev_brand" not in st.session_state or st.session_state["prev_brand"] != brand:
        st.session_state["prev_brand"] = brand
        st.session_state["model_idx"] = 0

    model_display = st.selectbox(
        "Model",
        all_model_display,
        index=st.session_state.get("model_idx", 0),
        key="model_select",
    )
    model_idx_now = all_model_display.index(model_display)
    model_name = all_model_raw[model_idx_now]
    is_catalog_only = model_display.endswith("(no listings in dataset)")
    is_other = model_name == "Other"

    if is_catalog_only:
        st.info(
            f"⚠️ **No listings for '{model_name}' in training data.** "
            "The estimate is based on brand and vehicle specifications and is less reliable."
        )
    if is_other:
        st.warning("⚠️ Unknown model — prediction uses brand-level patterns only.")

    # ── Auto-suggest defaults ─────────────────────────────────────
    model_defaults = _get_model_defaults(brand, model_name, comparable_df)

    with st.form("pred_form"):
        year = st.slider("Manufacturing Year", 2000, 2024, 2018)
        vehicle_age = 2024 - year

        km_driven = st.number_input(
            "Kilometres Driven", min_value=500, max_value=800000, value=45000, step=2500
        )

        c1, c2 = st.columns(2)
        with c1:
            fuel_idx = known_fuels.index(model_defaults["fuel"]) if model_defaults["fuel"] in known_fuels else 0
            fuel = st.selectbox("Fuel Type", known_fuels, index=fuel_idx)
        with c2:
            trans_idx = known_transmissions.index(model_defaults["transmission"]) \
                if model_defaults["transmission"] in known_transmissions else 0
            transmission = st.selectbox("Transmission", known_transmissions, index=trans_idx)

        c3, c4 = st.columns(2)
        with c3:
            engine_cc = st.number_input(
                "Engine (CC)", min_value=600, max_value=6000,
                value=int(model_defaults["engine_cc"]), step=50
            )
        with c4:
            max_power_bhp = st.number_input(
                "Max Power (bhp)", min_value=30.0, max_value=600.0,
                value=float(model_defaults["max_power_bhp"]), step=5.0
            )

        c5, c6 = st.columns(2)
        with c5:
            mileage_val = st.number_input(
                "Mileage (kmpl)", min_value=5.0, max_value=45.0,
                value=float(model_defaults["mileage_val"]), step=0.5
            )
        with c6:
            torque_nm = st.number_input(
                "Torque (Nm)", min_value=30.0, max_value=900.0,
                value=float(model_defaults["torque_nm"]), step=5.0
            )

        st.markdown("#### 👤 Ownership & Market Details")
        c7, c8 = st.columns(2)
        with c7:
            owner_str = st.selectbox("Owner History", known_owners)
        with c8:
            seller_type = st.selectbox("Seller Type", known_seller_types)

        c9, c10 = st.columns(2)
        with c9:
            seats = st.selectbox("Seats", [4, 5, 6, 7, 8, 9],
                                  index=1)
        with c10:
            trim_tier = st.selectbox("Trim Tier", ["Base", "Mid", "Top"], index=1)

        body_segment = st.selectbox(
            "Body Segment", ["Hatchback", "Sedan", "SUV", "MUV", "Van"], index=0
        )

        st.markdown("---")
        asking_price = st.number_input(
            "Seller Asking Price ₹ (optional)",
            min_value=0, max_value=100000000, value=0, step=10000,
        )

        submitted = st.form_submit_button("⚡ Estimate Resale Price")


# ════════════════════════════════════════════════════════════════════
# PREDICTION & OUTPUTS
# ════════════════════════════════════════════════════════════════════
with col_res:
    if submitted:
        owner_tier = encode_owner_tier(owner_str)
        effective_age = max(vehicle_age, 0.5)
        high_mileage_flag = 1 if (km_driven / effective_age) > 20000 else 0

        # Derived mileage features (must match feature_engineering.py exactly)
        km_per_year = round(km_driven / effective_age, 1)
        expected_km = effective_age * (typical_km_per_year if typical_km_per_year > 0 else 10000.0)
        km_ratio = float(np.clip(km_driven / max(expected_km, 1), 0.05, 10.0))

        user_input = {
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "km_per_year": km_per_year,
            "km_ratio": round(km_ratio, 4),
            "engine_cc": engine_cc,
            "mileage_val": mileage_val,
            "max_power_bhp": max_power_bhp,
            "owner_tier": owner_tier,
            "seats": float(seats),
            "torque_nm": torque_nm,
            "high_mileage_flag": high_mileage_flag,
            "brand": brand,
            "model_name": model_name if model_name != "Other" else "Unknown",
            "fuel": fuel,
            "transmission": transmission,
            "seller_type": seller_type,
            "trim_tier": trim_tier,
            "body_segment": body_segment,
        }

        input_df = _build_input_df(user_input, all_feature_cols)
        try:
            sample_vec = preprocessor.transform(input_df)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        predicted_price = max(0.0, float(best_model.predict(sample_vec)[0]))

        # Apply mileage adjustment (isotonic post-processing)
        adj_result = apply_mileage_adjustment(
            cart_price=predicted_price,
            km_ratio=km_ratio,
            adjuster=mileage_adjuster,
            enabled=mileage_adjuster is not None,
        )
        # Use adjusted price as the displayed fair-market value
        display_price = adj_result["final_price"]

        def _band(p):
            if p < 300000: return "Budget"
            elif p < 700000: return "Mid"
            elif p < 1500000: return "Premium"
            return "Luxury"
        price_band = _band(display_price)

        # ── 1. Fair Price + Range ────────────────────────────
        price_range = compute_price_range(
            display_price,
            calibration_residuals=calibration_residuals,
            price_band=price_band,
            residuals_by_band=residuals_by_band,
        )

        st.markdown("---")
        st.markdown("### 💰 Estimated Resale Price")

        if is_catalog_only or is_other:
            st.warning(
                "⚠️ **Low Reliability:** No training listings for this model. "
                "Estimate uses brand-level and specification-level patterns only."
            )

        c_low, c_fair, c_high = st.columns(3)
        c_low.metric("🔵 Conservative (10th pct)", format_inr(price_range["low"]))
        c_fair.metric("🎯 Fair Market Value", format_inr(price_range["fair"]))
        c_high.metric("🔴 Optimistic (90th pct)", format_inr(price_range["high"]))

        # Mileage adjustment breakdown
        if adj_result["enabled"]:
            factor = adj_result["adjustment_factor"]
            pct_adj = round((factor - 1.0) * 100, 1)
            arrow = "↓" if pct_adj < 0 else "↑"
            st.caption(
                f"🛣️ CART base: **{format_inr(adj_result['cart_base'])}** · "
                f"Mileage adjustment: **{arrow}{abs(pct_adj):.1f}%** "
                f"(km_ratio={km_ratio:.2f}) · "
                f"Final: **{format_inr(display_price)}**"
            )
        else:
            st.caption(
                "Range is derived from held-out test-set empirical residuals (not a statistical confidence interval)."
            )

        # ── 2. Reliability ───────────────────────────────────
        st.markdown("---")
        st.markdown("### 🛡️ Prediction Reliability")
        rel = evaluate_prediction_reliability(
            best_model, sample_vec, user_input,
            calibration_residuals=calibration_residuals,
            known_brands=set(meta.get("known_brands", [])),
        )
        icon = {"green": "🟢", "orange": "🟡", "red": "🔴"}.get(rel["support_color"], "⚪")
        st.info(f"{icon} {rel['explanation']}")
        for flag in rel["ood_flags"]:
            st.warning(f"⚠️ {flag}")

        # ── 3. Comparable Listings ───────────────────────────
        if not comparable_df.empty:
            st.markdown("---")
            st.markdown("### 📋 Comparable Real Listings")
            comps = get_comparable_listings(user_input, comparable_df, n=8)
            if not comps.empty:
                if "selling_price" in comps.columns:
                    comps["Price"] = comps["selling_price"].apply(format_inr)
                    comps = comps.drop(columns=["selling_price"])
                st.dataframe(comps, use_container_width=True)

        # ── 4. What-If ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔧 What-If Scenario Explorer")
        wi_c1, wi_c2 = st.columns(2)
        with wi_c1:
            wi_km = st.slider("Change: km Driven", 1000, 400000, int(km_driven), step=5000, key="wi_km")
            wi_age = st.slider("Change: Vehicle Age (yr)", 0, 20, int(vehicle_age), key="wi_age")
        with wi_c2:
            wi_owner = st.slider("Change: Owner Tier (1=First…4=Fourth+)", 1, 4, int(owner_tier), key="wi_owner")
            wi_trans = st.selectbox("Change: Transmission", ["Manual", "Automatic"],
                                     index=0 if transmission == "Manual" else 1, key="wi_trans")

        for label, factor, new_val in [
            (f"km_driven → {wi_km:,}", "km_driven", wi_km),
            (f"vehicle_age → {wi_age}y", "vehicle_age", wi_age),
            (f"owner_tier → {wi_owner}", "owner_tier", wi_owner),
            (f"transmission → {wi_trans}", "transmission", wi_trans),
        ]:
            r = compute_what_if(user_input, factor, new_val, best_model, preprocessor, all_feature_cols)
            d = "↑" if r["delta"] >= 0 else "↓"
            st.markdown(
                f"**{label}** → {format_inr(r['new_price'])} "
                f"({d} {format_inr(abs(r['delta']))}, {r['pct_delta']:+.1f}%)"
            )

        # ── 5. Depreciation Curve ────────────────────────────
        st.markdown("---")
        st.markdown("### 📉 Depreciation Curve")
        with st.spinner("Computing..."):
            curve = compute_depreciation_curve(user_input, best_model, preprocessor, all_feature_cols, max_age=20)
        if curve:
            ages = [pt["age"] for pt in curve]
            prices = [pt["predicted_price"] for pt in curve]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ages, y=prices, mode="lines+markers",
                                      line=dict(color="#4A90D9", width=3), marker=dict(size=5),
                                      hovertemplate="Age: %{x}y<br>Price: ₹%{y:,.0f}<extra></extra>"))
            fig.add_vline(x=vehicle_age, line_dash="dash", line_color="orange",
                           annotation_text=f"Now ({vehicle_age}y)")
            fig.update_layout(title="Price vs Age (all other factors fixed)",
                               xaxis_title="Vehicle Age (years)", yaxis_title="Price (₹)",
                               template="plotly_white", height=340)
            st.plotly_chart(fig, use_container_width=True)

        # ── 6. Factor Sensitivity ────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Factor Sensitivity (swap to training median)")
        sens = compute_factor_sensitivity(user_input, best_model, preprocessor, all_feature_cols, training_stats)
        if sens:
            rows = []
            for factor, vals in sorted(sens.items(), key=lambda x: -abs(x[1]["price_delta"])):
                rows.append({"Factor": factor, "Your Value": vals["base_val"],
                              "Median": vals["swapped_val"],
                              "Price Δ": format_inr(abs(vals["price_delta"])),
                              "Dir": "↑" if vals["price_delta"] >= 0 else "↓",
                              "% Impact": f"{vals['pct_delta']:+.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ── 6b. Mileage Impact Panel ──────────────────────────
        st.markdown("---")
        st.markdown("### 🛣️ Mileage Impact Analysis")
        st.caption(
            "How does resale price change as kilometres driven increases? "
            "All other factors held constant. "
            + ("Isotonic mileage adjustment applied." if mileage_adjuster is not None
               else "No mileage adjuster (retrain to enable).")
        )
        with st.spinner("Computing mileage impact..."):
            impact_pts = compute_mileage_impact_curve(
                base_input=user_input,
                model=best_model,
                preprocessor=preprocessor,
                feature_columns=all_feature_cols,
                adjuster=mileage_adjuster,
                typical_km_per_year=typical_km_per_year,
                n_points=25,
            )

        if impact_pts:
            km_vals = [p["km"] for p in impact_pts]
            cart_vals = [p["cart_price"] for p in impact_pts]
            adj_vals = [p["adjusted_price"] for p in impact_pts]

            fig_mi = go.Figure()
            fig_mi.add_trace(go.Scatter(
                x=km_vals, y=cart_vals, mode="lines",
                name="CART Base",
                line=dict(color="#4A90D9", width=2, dash="dot"),
                hovertemplate="km: %{x:,}<br>CART Base: ₹%{y:,.0f}<extra></extra>",
            ))
            if mileage_adjuster is not None:
                fig_mi.add_trace(go.Scatter(
                    x=km_vals, y=adj_vals, mode="lines+markers",
                    name="Adjusted",
                    line=dict(color="#E8742A", width=3),
                    marker=dict(size=4),
                    hovertemplate="km: %{x:,}<br>Adjusted: ₹%{y:,.0f}<extra></extra>",
                ))
            fig_mi.add_vline(
                x=km_driven, line_dash="dash", line_color="green",
                annotation_text=f"Your km ({km_driven:,})",
            )
            # Detect flat regions — same CART leaf
            n_unique_cart = len(set(cart_vals))
            if n_unique_cart <= 3:
                fig_mi.add_annotation(
                    text="⚠️ Same decision-tree leaf — CART price constant in this range",
                    xref="paper", yref="paper", x=0.5, y=0.95,
                    showarrow=False, font=dict(color="red", size=12),
                )
            fig_mi.update_layout(
                title="Price vs Kilometres Driven (all other factors fixed)",
                xaxis_title="km Driven",
                yaxis_title="Price (₹)",
                template="plotly_white",
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_mi, use_container_width=True)

            # Mileage impact table
            checkpoints = [0.25, 0.50, 1.0, 1.50, 2.0, 3.0]
            table_rows = []
            for mult in checkpoints:
                target_km = int(km_driven * mult)
                # Find closest point
                closest = min(impact_pts, key=lambda p: abs(p["km"] - target_km))
                adj_p = closest["adjusted_price"] if mileage_adjuster is not None else closest["cart_price"]
                base_p = impact_pts[len(impact_pts)//2]["adjusted_price"] if mileage_adjuster is not None \
                    else impact_pts[len(impact_pts)//2]["cart_price"]
                pct = round((adj_p / base_p - 1) * 100, 1) if base_p > 0 else 0.0
                table_rows.append({
                    "km Driven": f"{target_km:,}",
                    "× Your km": f"{mult:.2f}×",
                    "Est. Price": format_inr(adj_p),
                    "vs 1× km": f"{pct:+.1f}%",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            if n_unique_cart <= 3:
                st.warning(
                    "⚠️ **Same decision-tree leaf:** The CART model assigns the same price "
                    "across the full km range shown above — the tree has no split on km for this car. "
                    "The mileage adjustment (orange line) corrects for this using the isotonic layer."
                    if mileage_adjuster is not None else
                    "⚠️ **Same decision-tree leaf:** No km splits reached for this car's profile. "
                    "Retrain to enable the mileage adjustment layer."
                )

        # ── 7. Decision Path ─────────────────────────────────
        st.markdown("---")
        with st.expander("🧭 Decision Path Explanation", expanded=False):
            expl = explain_prediction_path(best_model, feat_names, sample_vec, user_input)
            st.write(expl["summary"])
            for step in expl["narrative"]:
                st.markdown(f"- {step}")
            if adj_result["enabled"]:
                st.markdown(
                    f"🛣️ *Mileage adjustment applied after CART prediction: "
                    f"factor = {adj_result['adjustment_factor']:.4f} "
                    f"(km_ratio = {km_ratio:.3f})*"
                )

        # ── 8. Market Positioning ────────────────────────────
        if asking_price > 0:
            st.markdown("---")
            st.markdown("### 🏷️ Market Positioning")
            pos = compute_market_positioning(display_price, asking_price)
            if pos:
                badge = {"green": "🟢", "red": "🔴", "blue": "🔵"}.get(pos["status_color"], "⚪")
                st.markdown(
                    f"**{badge} {pos['positioning']}** "
                    f"(Asking: {format_inr(pos['asking_price'])} · "
                    f"Estimate: {format_inr(pos['estimated_price'])} · "
                    f"{pos['diff_pct']:+.1f}%)"
                )
                st.caption(pos["recommendation"])

    else:
        st.info("👈 Select brand/model, fill vehicle details, then click **⚡ Estimate Resale Price**.")
        st.markdown("""
        **Features used:**
        - 🕰️ Vehicle age & km intensity  · ⚙️ Engine, power, torque, mileage
        - 👤 Ownership tier & seller type · 🏷️ Trim (Base/Mid/Top) & body segment
        - 🚗 Brand & model (from full dataset — all models available)

        **Not in this dataset:** 📍 City/Location · 💥 Accident history · 🛠️ Service records
        """)
