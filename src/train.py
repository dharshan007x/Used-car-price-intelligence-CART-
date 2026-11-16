"""
Master Model Training Pipeline for Used Car Price Intelligence.

Executes end-to-end data validation, feature engineering, train/validation/test splitting,
depth experimentation, cost-complexity pruning, ablation analysis, final test evaluation,
feature importance reporting, and model artifact persistence.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from datetime import datetime

from src.data_loader import load_raw_dataset, DEFAULT_PROCESSED_PATH
from src.data_validation import perform_data_validation
from src.feature_engineering import engineer_features, build_brand_model_map
from src.preprocessing import (
    build_preprocessor,
    get_feature_names_from_preprocessor,
    save_pipeline,
    MARKET_NUMERICAL_FEATURES,
    MARKET_CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from src.cart_model import create_cart_regressor, evaluate_cart_performance, extract_tree_properties, build_monotonic_cst
from src.depth_analysis import run_depth_experiments
from src.pruning import run_pruning_experiments
from src.evaluation import compute_comprehensive_metrics, compute_segment_metrics, build_model_comparison_table
from src.ablation import run_ablation_study
from src.prediction import fit_mileage_adjuster

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ── Factor group mapping for importance aggregation ──────────────────
FACTOR_GROUPS: Dict[str, List[str]] = {
    "Age": ["vehicle_age"],
    "Usage": ["km_driven", "km_per_year", "km_ratio", "high_mileage_flag"],
    "Engine & Power": ["engine_cc", "mileage_val", "max_power_bhp", "torque_nm"],
    "Brand": ["brand"],
    "Model": ["model_name"],
    "Fuel & Transmission": ["fuel", "transmission"],
    "Ownership": ["owner_tier", "seller_type"],
    "Trim & Segment": ["trim_tier", "body_segment"],
    "Comfort": ["seats"],
}


def _get_factor_group(feature_name: str) -> str:
    """Maps a (possibly OHE-prefixed) feature name to its factor group."""
    # Strip OHE transformer prefix e.g. 'cat_hc__brand_Maruti' -> 'brand_Maruti'
    stripped = feature_name
    for pfx in ["num__", "cat_hc__", "cat_std__"]:
        if feature_name.startswith(pfx):
            stripped = feature_name[len(pfx):]
            break
    for group, cols in FACTOR_GROUPS.items():
        for col in cols:
            if stripped == col or stripped.startswith(col + "_"):
                return group
    return "Other"


def _compute_price_band(price: float) -> str:
    if price < 300000:
        return "Budget"
    elif price < 700000:
        return "Mid"
    elif price < 1500000:
        return "Premium"
    else:
        return "Luxury"


def _build_feature_availability_report(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Generates reports/feature_availability.csv with one row per factor.
    """
    rows = []
    cols = df_raw.columns.tolist()

    def _check(raw_col, factor, group, derived=False, notes=""):
        if raw_col in cols:
            pct_missing = round(df_raw[raw_col].isnull().mean() * 100, 2)
            n_unique = df_raw[raw_col].nunique(dropna=True)
            enabled = pct_missing < 50
        else:
            pct_missing = 100.0
            n_unique = 0
            enabled = derived  # derived features may still be computed
        rows.append({
            "factor_group": group,
            "column_name": raw_col,
            "available": raw_col in cols or derived,
            "derived": derived,
            "pct_missing": pct_missing if raw_col in cols else "N/A",
            "n_unique": n_unique if raw_col in cols else "N/A",
            "enabled": enabled,
            "notes": notes,
        })

    _check("year", "year", "Age", notes="Used to derive vehicle_age")
    _check("vehicle_age", "vehicle_age", "Age", derived=True, notes="Derived: reference_year - year")
    _check("km_driven", "km_driven", "Usage")
    _check("high_mileage_flag", "high_mileage_flag", "Usage", derived=True, notes="Derived: km_per_year > 20000")
    _check("fuel", "fuel", "Fuel & Transmission")
    _check("transmission", "transmission", "Fuel & Transmission")
    _check("engine", "engine", "Engine & Power", notes="Raw; parsed to engine_cc")
    _check("engine_cc", "engine_cc", "Engine & Power", derived=True, notes="Parsed from engine column")
    _check("mileage", "mileage", "Engine & Power", notes="Raw; parsed to mileage_val")
    _check("max_power", "max_power", "Engine & Power", notes="Raw; parsed to max_power_bhp")
    _check("torque", "torque", "Engine & Power", notes="Raw; parsed to torque_nm")
    _check("owner", "owner", "Ownership", notes="Encoded as owner_tier (1-5)")
    _check("seller_type", "seller_type", "Ownership")
    _check("seats", "seats", "Comfort")
    _check("trim_tier", "trim_tier", "Trim & Segment", derived=True, notes="Derived from name variant tokens")
    _check("body_segment", "body_segment", "Trim & Segment", derived=True, notes="Derived from model lookup table")
    _check("name", "name", "Brand", notes="Source for brand, model_name, trim_tier extraction")
    _check("brand", "brand", "Brand", derived=True, notes="Extracted from name")
    _check("model_name", "model_name", "Model", derived=True, notes="Extracted from name")
    # Explicitly absent features
    rows.append({
        "factor_group": "Location",
        "column_name": "city",
        "available": False, "derived": False,
        "pct_missing": "N/A", "n_unique": "N/A",
        "enabled": False,
        "notes": "NOT IN DATASET. Add 'India Used Car Dataset with City' from Kaggle for location features.",
    })
    rows.append({
        "factor_group": "Condition",
        "column_name": "accident_history",
        "available": False, "derived": False,
        "pct_missing": "N/A", "n_unique": "N/A",
        "enabled": False,
        "notes": "NOT IN DATASET. Not available in CarDekho v3 dataset.",
    })
    return pd.DataFrame(rows)


def _build_model_coverage_audit(
    df_full: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    brand_model_map: Dict,
) -> pd.DataFrame:
    """
    Generates reports/model_coverage_audit.csv showing every (brand, model)
    pair and whether it appears in the dropdown.
    """
    rows = []
    all_pairs = df_full.groupby(["brand", "model_name"]).size().reset_index(name="raw_count")
    train_counts = X_train.groupby(["brand", "model_name"]).size().reset_index(name="train_count") \
        if "brand" in X_train.columns and "model_name" in X_train.columns else pd.DataFrame()
    test_counts = X_test.groupby(["brand", "model_name"]).size().reset_index(name="test_count") \
        if "brand" in X_test.columns and "model_name" in X_test.columns else pd.DataFrame()

    # Build lookup sets
    dropdown_pairs = set()
    for brand, models in brand_model_map.items():
        for m in models:
            dropdown_pairs.add((brand, m["name"]))

    for _, row in all_pairs.iterrows():
        brand, model = str(row["brand"]), str(row["model_name"])
        raw_count = int(row["raw_count"])

        tc = train_counts[
            (train_counts["brand"] == brand) & (train_counts["model_name"] == model)
        ]["train_count"].sum() if not train_counts.empty else 0

        tesc = test_counts[
            (test_counts["brand"] == brand) & (test_counts["model_name"] == model)
        ]["test_count"].sum() if not test_counts.empty else 0

        in_dropdown = (brand, model) in dropdown_pairs
        reason = "" if in_dropdown else "Should be in map — investigate"

        rows.append({
            "brand": brand,
            "model": model,
            "raw_count": raw_count,
            "train_count": int(tc),
            "test_count": int(tesc),
            "in_dropdown": in_dropdown,
            "reason_if_missing": reason,
        })

    audit_df = pd.DataFrame(rows).sort_values(["brand", "model"])
    return audit_df


def run_full_training_pipeline(random_state: int = 42) -> Dict[str, Any]:
    """
    Executes the end-to-end market-grade training and evaluation pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Raw Data ──────────────────────────────────────────
    logger.info("Step 1: Loading raw dataset...")
    df_raw = load_raw_dataset()

    # ── Feature Availability Report ───────────────────────────────
    avail_df = _build_feature_availability_report(df_raw)
    avail_df.to_csv(REPORTS_DIR / "feature_availability.csv", index=False)
    logger.info("Saved reports/feature_availability.csv")

    # ── 2. Validate and Clean Data ────────────────────────────────
    logger.info("Step 2: Validating and cleaning data...")
    df_clean, audit_report = perform_data_validation(df_raw, reference_year=2024)

    # ── 3. Feature Engineering (two-pass, leak-free) ─────────────
    # Pass 1: engineer WITHOUT typical_km_per_year (km_ratio defaults to 1.0).
    # This is only used to establish the train/test split so we can compute
    # typical_km_per_year from training rows only. No km_ratio in pass-1.
    logger.info("Step 3a: First-pass feature engineering (for split)...")
    df_eng_pass1 = engineer_features(df_clean, reference_year=2024, typical_km_per_year=None)

    # Determine feature cols available in pass-1 (km_ratio present but = 1.0 placeholder)
    _num_pass1 = [c for c in MARKET_NUMERICAL_FEATURES if c in df_eng_pass1.columns]
    _cat_pass1 = [c for c in MARKET_CATEGORICAL_FEATURES if c in df_eng_pass1.columns]
    _all_pass1 = _num_pass1 + _cat_pass1

    _valid_pass1 = df_eng_pass1.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    _X_p1 = _valid_pass1[_all_pass1]
    _y_p1 = _valid_pass1[TARGET_COLUMN].values

    # Same split indices as full run (random_state must match)
    _X_train_p1, _X_test_p1, _, _ = train_test_split(_X_p1, _y_p1, test_size=0.20, random_state=random_state)
    _X_train_sub_p1, _, _, _ = train_test_split(_X_train_p1, _y_p1[:len(_X_train_p1)], test_size=0.25, random_state=random_state)

    # Compute typical_km_per_year on training partition ONLY (no leakage)
    _train_valid = _valid_pass1.loc[_X_train_p1.index]
    if "km_per_year" in _train_valid.columns:
        typical_km_per_year = float(_train_valid["km_per_year"].median())
    elif "km_driven" in _train_valid.columns and "vehicle_age" in _train_valid.columns:
        _eff_age = _train_valid["vehicle_age"].clip(lower=0.5)
        typical_km_per_year = float((_train_valid["km_driven"] / _eff_age).median())
    else:
        typical_km_per_year = 10000.0
    logger.info(f"Computed typical_km_per_year from training data: {typical_km_per_year:.1f} km/year")

    # Pass 2: re-engineer with correct typical_km_per_year → km_ratio now meaningful
    logger.info("Step 3b: Second-pass feature engineering (with km_ratio)...")
    df_eng = engineer_features(df_clean, reference_year=2024, typical_km_per_year=typical_km_per_year)

    # ── Build Brand-Model Map from FULL cleaned dataset ───────────
    # This map is for UI dropdowns ONLY. It is built BEFORE the train/test split
    # so no test-split models are lost. It does not affect training in any way.
    logger.info("Step 3b: Building brand-model map from full cleaned dataset...")
    brand_model_map = build_brand_model_map(df_eng, brand_col="brand", model_col="model_name")
    brand_model_map_path = MODELS_DIR / "brand_model_map.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    map_meta = {
        "_description": "Brand->Model map for UI dropdowns. Built from FULL cleaned dataset (not just train split). No min_frequency filter applied.",
        "_created_at": datetime.now().isoformat(),
        "_source": "Full cleaned CarDekho v3 dataset",
        "_n_brands": len(brand_model_map),
        "_n_models": sum(len(v) for v in brand_model_map.values()),
        "brands": brand_model_map,
    }
    with open(brand_model_map_path, "w", encoding="utf-8") as f:
        json.dump(map_meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved brand_model_map.json: {len(brand_model_map)} brands, {map_meta['_n_models']} models")

    # Save processed data
    DEFAULT_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_eng.to_csv(DEFAULT_PROCESSED_PATH, index=False)
    logger.info(f"Saved processed dataset to {DEFAULT_PROCESSED_PATH}")

    # ── 4. Build feature lists from available columns ─────────────
    num_cols = [c for c in MARKET_NUMERICAL_FEATURES if c in df_eng.columns]
    cat_cols = [c for c in MARKET_CATEGORICAL_FEATURES if c in df_eng.columns]
    all_feature_cols = num_cols + cat_cols

    logger.info(f"Market features: {len(num_cols)} numerical + {len(cat_cols)} categorical")

    # ── 5. Train-Test Split ───────────────────────────────────────
    logger.info("Step 4: Splitting 80% Train / 20% Test...")
    valid_data = df_eng.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    # Fill seats NaN with median before split
    if "seats" in valid_data.columns:
        valid_data["seats"] = pd.to_numeric(valid_data["seats"], errors="coerce")

    X = valid_data[all_feature_cols]
    y = valid_data[TARGET_COLUMN].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=random_state
    )
    logger.info(f"Splits — TrainSub={len(X_train_sub)}, Val={len(X_val)}, Test={len(X_test)}")

    # ── Coverage Audit: which (brand, model) pairs appear where ──
    # We use valid_data (the full model-eligible set) and the split indices
    audit_cols = ["brand", "model_name"]
    has_audit_cols = all(c in valid_data.columns for c in audit_cols)
    if has_audit_cols:
        X_train_aug = valid_data.loc[X_train_full.index, audit_cols]
        X_test_aug = valid_data.loc[X_test.index, audit_cols]
    else:
        X_train_aug = pd.DataFrame()
        X_test_aug = pd.DataFrame()
    coverage_df = _build_model_coverage_audit(df_eng, X_train_aug, X_test_aug, brand_model_map)
    coverage_df.to_csv(REPORTS_DIR / "model_coverage_audit.csv", index=False)
    n_missing = coverage_df[~coverage_df["in_dropdown"]].shape[0]
    logger.info(f"Model coverage audit: {len(coverage_df)} (brand,model) pairs, {n_missing} missing from dropdown")

    # ── 6. Fit Preprocessor on Train Sub Only ─────────────────────
    logger.info("Step 5: Fitting preprocessor on training data only...")
    preprocessor = build_preprocessor(num_features=num_cols, cat_features=cat_cols)
    X_train_sub_trans = preprocessor.fit_transform(X_train_sub)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names_from_preprocessor(preprocessor)
    logger.info(f"Transformed feature dimension: {len(feature_names)}")

    # ── 7. Depth Experimentation ──────────────────────────────────
    logger.info("Step 6: Running tree depth experiments...")
    depth_df, depth_interp = run_depth_experiments(
        X_train_sub_trans, y_train_sub, X_val_trans, y_val, random_state=random_state
    )
    depth_df.to_csv(REPORTS_DIR / "depth_analysis.csv", index=False)
    logger.info(f"Best Depth (Val RMSE): {depth_interp['best_depth']}")

    # ── 8. Pruning Experimentation ────────────────────────────────
    logger.info("Step 7: Running cost-complexity pruning experiments...")
    pruning_df, pruning_interp, _ = run_pruning_experiments(
        X_train_sub_trans, y_train_sub, X_val_trans, y_val, random_state=random_state
    )
    pruning_df.to_csv(REPORTS_DIR / "pruning_analysis.csv", index=False)
    logger.info(f"Best Alpha (Val RMSE): {pruning_interp['best_alpha']}")

    # ── 9. Ablation Study ────────────────────────────────────────
    logger.info("Step 8: Running ablation study (4 feature sets)...")
    ablation_df, _ = run_ablation_study(
        X_train_sub, y_train_sub,
        X_val, y_val,
        X_test, y_test,
        depth=depth_interp["best_depth"],
        random_state=random_state,
    )
    ablation_df.to_csv(REPORTS_DIR / "ablation_analysis.csv", index=False)
    logger.info("Saved reports/ablation_analysis.csv")

    # ── 10. Retrain on Full 80% ───────────────────────────────────
    logger.info("Step 9: Retraining on full 80% training set...")
    preprocessor_full = build_preprocessor(num_features=num_cols, cat_features=cat_cols)
    X_train_full_trans = preprocessor_full.fit_transform(X_train_full)
    X_test_final_trans = preprocessor_full.transform(X_test)
    full_feature_names = get_feature_names_from_preprocessor(preprocessor_full)

    best_d = depth_interp["best_depth"]
    best_a = pruning_interp["best_alpha"]

    # Build monotonic constraints aligned to post-transform feature names
    mono_cst = build_monotonic_cst(full_feature_names)
    logger.info(f"Monotonic constraints: {sum(v != 0 for v in mono_cst)} constrained features "
                f"(km_driven, km_per_year, km_ratio, vehicle_age, owner_tier → decreasing)")

    unrestricted_model = create_cart_regressor(max_depth=None, random_state=random_state,
                                               monotonic_cst=mono_cst)
    unrestricted_model.fit(X_train_full_trans, y_train_full)

    depth_restricted_model = create_cart_regressor(max_depth=best_d, random_state=random_state,
                                                   monotonic_cst=mono_cst)
    depth_restricted_model.fit(X_train_full_trans, y_train_full)

    pruned_model = create_cart_regressor(ccp_alpha=best_a, random_state=random_state,
                                         monotonic_cst=mono_cst)
    pruned_model.fit(X_train_full_trans, y_train_full)

    linear_baseline = LinearRegression()
    linear_baseline.fit(X_train_full_trans, y_train_full)

    candidate_models = {
        "Unrestricted CART Tree": unrestricted_model,
        "Depth-Restricted CART Tree": depth_restricted_model,
        "Cost-Complexity Pruned CART Tree": pruned_model,
        "Linear Regression (Baseline)": linear_baseline,
    }

    # ── 11. Final Test Evaluation ─────────────────────────────────
    logger.info("Step 10: Evaluating on untouched test set...")
    model_comparison_df = build_model_comparison_table(candidate_models, X_test_final_trans, y_test)
    model_comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    cart_only = model_comparison_df[model_comparison_df["Model"].str.contains("CART")]
    best_cart_row = cart_only.loc[cart_only["RMSE (₹)"].idxmin()]
    best_model_name = str(best_cart_row["Model"])
    best_cart_model = candidate_models[best_model_name]
    logger.info(f"Selected primary model: {best_model_name}")

    # ── 11a. km Sensitivity Audit CSV ────────────────────────────
    logger.info("Step 10a: Computing km sensitivity audit...")
    try:
        from scipy import stats as _stats
        km_col = "km_driven" if "km_driven" in X_test.columns else None
        if km_col:
            y_test_pred_raw = best_cart_model.predict(X_test_final_trans)
            # Build audit by cohort (brand+model) on test set
            _test_aug = valid_data.loc[X_test.index, [c for c in ["brand", "model_name", "vehicle_age", "km_driven"]
                                                       if c in valid_data.columns]].copy()
            _test_aug["actual"] = y_test
            _test_aug["pred"] = y_test_pred_raw

            audit_rows = []
            # Overall
            r_all, p_all = _stats.spearmanr(_test_aug["km_driven"], _test_aug["actual"])
            audit_rows.append({"brand": "__ALL__", "model": "__ALL__", "age_band": "all",
                                "n": len(_test_aug), "spearman_r_km_actual": round(float(r_all), 3),
                                "p_value": round(float(p_all), 4)})
            # Per (brand, model, age_band)
            if "brand" in _test_aug.columns and "model_name" in _test_aug.columns:
                _test_aug["age_band"] = pd.cut(_test_aug["vehicle_age"],
                                                bins=[0, 3, 7, 12, 100],
                                                labels=["0-3y", "3-7y", "7-12y", "12+y"])
                for (brand, model, ab), grp in _test_aug.groupby(["brand", "model_name", "age_band"]):
                    if len(grp) < 5:
                        continue
                    r, p = _stats.spearmanr(grp["km_driven"], grp["actual"])
                    audit_rows.append({"brand": brand, "model": model, "age_band": str(ab),
                                       "n": len(grp), "spearman_r_km_actual": round(float(r), 3),
                                       "p_value": round(float(p), 4)})
            pd.DataFrame(audit_rows).to_csv(REPORTS_DIR / "km_sensitivity_audit.csv", index=False)
            logger.info("Saved reports/km_sensitivity_audit.csv")
    except Exception as _e:
        logger.warning(f"km audit skipped: {_e}")

    # ── 11b. Fit Mileage Adjustment Layer on Validation Partition ──
    # Uses validation set (not test) to avoid leakage. The adjuster maps
    # km_ratio → residual_ratio = actual/cart_pred (isotonic, decreasing).
    logger.info("Step 10b: Fitting isotonic mileage adjuster on validation data...")
    mileage_adjuster = None
    adj_meta: Dict[str, Any] = {"enabled": False, "reason": "km_ratio not in features"}
    if "km_ratio" in num_cols:
        # Refit preprocessor on train_sub to transform val set
        _pre_adj = build_preprocessor(num_features=num_cols, cat_features=cat_cols)
        _pre_adj.fit(X_train_sub)
        X_val_adj_trans = _pre_adj.transform(X_val)
        # Build constraints aligned to _pre_adj's feature count (may differ from full)
        _adj_feat_names = get_feature_names_from_preprocessor(_pre_adj)
        _adj_mono_cst = build_monotonic_cst(_adj_feat_names)
        # Get val predictions from a sub-fitted model using sub-preprocessor
        _adj_model = create_cart_regressor(max_depth=best_d, random_state=random_state,
                                           monotonic_cst=_adj_mono_cst)
        _adj_model.fit(_pre_adj.transform(X_train_sub), y_train_sub)
        val_cart_pred = _adj_model.predict(X_val_adj_trans)

        # Extract km_ratio values for validation rows
        km_ratio_col_idx = num_cols.index("km_ratio") if "km_ratio" in num_cols else -1
        if km_ratio_col_idx >= 0 and "km_ratio" in X_val.columns:
            val_km_ratio = X_val["km_ratio"].fillna(1.0).values
            mileage_adjuster = fit_mileage_adjuster(val_km_ratio, y_val, val_cart_pred)
            save_pipeline(mileage_adjuster, MODELS_DIR / "mileage_adjuster.joblib")
            adj_meta = {
                "enabled": True,
                "adj_lower": 0.80,
                "adj_upper": 1.15,
                "n_val_samples": len(y_val),
                "km_ratio_col_idx": km_ratio_col_idx,
            }
            logger.info("Saved mileage_adjuster.joblib")
        else:
            logger.warning("km_ratio column not found in validation data — adjuster skipped")
    else:
        logger.warning("km_ratio not in num_cols — mileage adjuster not fitted")

    y_test_pred = best_cart_model.predict(X_test_final_trans)
    final_test_metrics = compute_comprehensive_metrics(y_test, y_test_pred, dataset_name="Test Set")
    segment_df = compute_segment_metrics(y_test, y_test_pred)
    segment_df.to_csv(REPORTS_DIR / "segment_performance.csv", index=False)

    # ── 12. Calibration Residuals by Price Band ───────────────────
    test_residuals = (y_test - y_test_pred).tolist()
    residuals_by_band: Dict[str, List[float]] = {"Budget": [], "Mid": [], "Premium": [], "Luxury": []}
    for actual, pred in zip(y_test, y_test_pred):
        band = _compute_price_band(float(actual))
        residuals_by_band[band].append(float(actual) - float(pred))

    # ── 13. Feature Importance Report ────────────────────────────
    logger.info("Step 11: Computing feature importance by factor group...")
    importances = best_cart_model.feature_importances_
    total = importances.sum()
    group_importance: Dict[str, float] = {}
    for feat_name, imp in zip(full_feature_names, importances):
        group = _get_factor_group(feat_name)
        group_importance[group] = group_importance.get(group, 0.0) + imp

    fi_rows = []
    for group, imp in sorted(group_importance.items(), key=lambda x: -x[1]):
        fi_rows.append({
            "factor_group": group,
            "total_importance": round(imp, 6),
            "pct_of_total": round(imp / total * 100, 2) if total > 0 else 0.0,
        })
    fi_df = pd.DataFrame(fi_rows)
    fi_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    logger.info("Saved reports/feature_importance.csv")

    # Dominance check
    max_pct = fi_df["pct_of_total"].max() if not fi_df.empty else 0.0
    dominant_group = fi_df.loc[fi_df["pct_of_total"].idxmax(), "factor_group"] if not fi_df.empty else "Unknown"
    dominance_warning = bool(max_pct > 70.0)
    if dominance_warning:
        logger.warning(f"Model dominance detected: '{dominant_group}' accounts for {max_pct:.1f}% of importance.")

    # ── 14. Comparable Records for Prediction Page ───────────────
    comparable_cols = [c for c in [
        "brand", "model_name", "year", "vehicle_age", "km_driven",
        "fuel", "transmission", "owner", "seller_type", "selling_price"
    ] if c in valid_data.columns]
    # Save training partition only (avoid test leakage)
    train_indices = X_train_full.index
    comparable_df = valid_data.loc[train_indices, comparable_cols].reset_index(drop=True)
    comparable_df.to_csv(MODELS_DIR / "comparable_records.csv", index=False)
    logger.info("Saved models/comparable_records.csv")

    # Training stats for sensitivity analysis
    training_stats = {
        "medians": {
            col: float(X_train_full[col].median())
            for col in num_cols if col in X_train_full.columns
        },
        "typical_km_per_year": typical_km_per_year,
    }

    # ── 15. Save Models and Metadata ────────────────────────────
    logger.info("Step 12: Saving model artifacts...")
    save_pipeline(best_cart_model, MODELS_DIR / "best_cart_model.joblib")
    save_pipeline(preprocessor_full, MODELS_DIR / "preprocessing_pipeline.joblib")
    save_pipeline(unrestricted_model, MODELS_DIR / "unrestricted_cart_model.joblib")
    save_pipeline(depth_restricted_model, MODELS_DIR / "depth_restricted_cart_model.joblib")
    save_pipeline(pruned_model, MODELS_DIR / "pruned_cart_model.joblib")
    save_pipeline(linear_baseline, MODELS_DIR / "linear_baseline_model.joblib")

    metadata = {
        "dataset_name": "CarDekho Used Car Dataset v3",
        "reference_year": 2024,
        "total_records_initial": audit_report["initial_record_count"],
        "records_after_cleaning": audit_report["final_record_count"],
        "train_samples": len(X_train_full),
        "test_samples": len(X_test),
        "target_variable": TARGET_COLUMN,
        "input_features": all_feature_cols,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "transformed_feature_names": full_feature_names,
        "known_brands": sorted(list(valid_data["brand"].dropna().unique())) if "brand" in valid_data.columns else [],
        "known_models": sorted(list(valid_data["model_name"].dropna().unique())) if "model_name" in valid_data.columns else [],
        "known_fuels": sorted(list(valid_data["fuel"].dropna().unique())) if "fuel" in valid_data.columns else [],
        "known_transmissions": sorted(list(valid_data["transmission"].dropna().unique())) if "transmission" in valid_data.columns else [],
        "known_seller_types": sorted(list(valid_data["seller_type"].dropna().unique())) if "seller_type" in valid_data.columns else [],
        "known_owners": sorted(list(valid_data["owner"].dropna().unique())) if "owner" in valid_data.columns else [],
        "brand_model_map_meta": {
            "created_at": datetime.now().isoformat(),
            "source": "Full cleaned dataset (train + test combined)",
            "n_brands": len(brand_model_map),
            "n_models": sum(len(v) for v in brand_model_map.values()),
            "note": "Built from full cleaned dataset — no min_frequency filter. For UI dropdowns only.",
        },
        "best_model_name": best_model_name,
        "best_max_depth": best_d,
        "best_ccp_alpha": best_a,
        "final_test_metrics": final_test_metrics,
        "calibration_residuals_sample": test_residuals[:300],
        "calibration_residuals_by_band": {
            band: vals[:100] for band, vals in residuals_by_band.items()
        },
        "tree_properties": extract_tree_properties(best_cart_model),
        "depth_interpretation": depth_interp,
        "pruning_interpretation": pruning_interp,
        "feature_importance_by_group": fi_df.to_dict(orient="records"),
        "dominance_warning": dominance_warning,
        "dominant_group": dominant_group,
        "dominant_group_pct": round(float(max_pct), 2),
        "training_stats": training_stats,
        "typical_km_per_year": typical_km_per_year,
        "mileage_adjuster": adj_meta,
        "unavailable_factors": [
            "city/location — NOT IN DATASET (add 'India Used Car Dataset with City' from Kaggle)",
            "accident_history — NOT IN DATASET",
            "service_history — NOT IN DATASET",
            "colour — NOT IN DATASET",
            "tyre_condition — NOT IN DATASET",
        ],
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    summary_json = {
        "metadata": metadata,
        "audit_report": audit_report,
        "model_comparison": model_comparison_df.to_dict(orient="records"),
        "segment_performance": segment_df.to_dict(orient="records"),
        "ablation_analysis": ablation_df.to_dict(orient="records"),
        "feature_importance": fi_df.to_dict(orient="records"),
    }

    with open(REPORTS_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary_json, f, indent=4)

    logger.info("✅ Training pipeline completed successfully.")
    return summary_json


if __name__ == "__main__":
    run_full_training_pipeline()
