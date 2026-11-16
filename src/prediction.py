"""
Prediction Pipeline and Market Intelligence Module for Used Car Price Intelligence.

Handles user input formatting, model scoring, empirical prediction intervals,
comparable listings, depreciation curves, factor sensitivity, what-if analysis,
out-of-distribution warnings, seller asking price market positioning,
and mileage adjustment (isotonic regression post-processing layer).
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression


# ══════════════════════════════════════════════════════════════════
# Formatting Utilities
# ══════════════════════════════════════════════════════════════════

def format_inr(amount: float) -> str:
    """Formats a number into standard Indian numbering format (₹X.XX Lakh / ₹X.XX Cr)."""
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return "₹0"

    val = round(amount)
    if val >= 10000000:
        return f"₹{val / 10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val / 100000:.2f} Lakh"
    else:
        s = str(abs(val))
        if len(s) <= 3:
            res = s
        else:
            last3 = s[-3:]
            rest = s[:-3]
            groups = []
            while len(rest) > 2:
                groups.insert(0, rest[-2:])
                rest = rest[:-2]
            if rest:
                groups.insert(0, rest)
            res = ",".join(groups) + "," + last3
        return f"₹{res}" if val >= 0 else f"-₹{res}"


# ══════════════════════════════════════════════════════════════════
# Price Range from Calibration Residuals
# ══════════════════════════════════════════════════════════════════

def compute_price_range(
    predicted_price: float,
    calibration_residuals: Optional[np.ndarray] = None,
    price_band: Optional[str] = None,
    residuals_by_band: Optional[Dict[str, np.ndarray]] = None,
    p_low: float = 10.0,
    p_high: float = 90.0,
) -> Dict[str, float]:
    """
    Returns empirical price range {low, fair, high} using held-out residuals.

    Uses band-specific residuals when available, otherwise global residuals.
    low  = max(0, predicted + p10-percentile residual)
    fair = predicted
    high = predicted + p90-percentile residual
    """
    residuals = None
    if residuals_by_band and price_band and price_band in residuals_by_band:
        residuals = np.array(residuals_by_band[price_band])

    if residuals is None and calibration_residuals is not None and len(calibration_residuals) > 0:
        residuals = np.array(calibration_residuals)

    if residuals is not None and len(residuals) > 5:
        low_delta = float(np.percentile(residuals, p_low))
        high_delta = float(np.percentile(residuals, p_high))
    else:
        # Fallback: ±15% symmetric heuristic
        low_delta = -predicted_price * 0.15
        high_delta = predicted_price * 0.15

    low = max(0.0, predicted_price + low_delta)
    high = predicted_price + high_delta

    return {"low": low, "fair": predicted_price, "high": high}


# ══════════════════════════════════════════════════════════════════
# Comparable Listings from Training Data
# ══════════════════════════════════════════════════════════════════

def get_comparable_listings(
    user_input: Dict[str, Any],
    training_df: pd.DataFrame,
    n: int = 8,
) -> pd.DataFrame:
    """
    Returns up to n real training rows that are most similar to the user's car.

    Similarity score = |age_diff| + |km_diff| / 10000.
    Prefers same brand+model, falls back to same brand only.
    """
    df = training_df.copy()

    brand = user_input.get("brand", None)
    model = user_input.get("model_name", None)
    user_age = user_input.get("vehicle_age", 5)
    user_km = user_input.get("km_driven", 50000)
    fuel = user_input.get("fuel", None)

    # Filter: same brand+model if columns exist
    subset = df.copy()
    if brand and "brand" in df.columns:
        same_bm = df[df["brand"] == brand]
        if model and "model_name" in df.columns:
            same_bm_m = same_bm[same_bm["model_name"] == model]
            if len(same_bm_m) >= 3:
                subset = same_bm_m
            else:
                subset = same_bm if len(same_bm) >= 3 else df
        else:
            subset = same_bm if len(same_bm) >= 3 else df

    # Similarity score
    age_col = "vehicle_age" if "vehicle_age" in subset.columns else None
    km_col = "km_driven" if "km_driven" in subset.columns else None

    if age_col and km_col:
        subset = subset.copy()
        subset["_sim"] = (
            (subset[age_col] - user_age).abs() +
            (subset[km_col] - user_km).abs() / 10000
        )
        subset = subset.sort_values("_sim").drop(columns=["_sim"])
    
    # Pick columns to display
    display_cols = []
    for c in ["brand", "model_name", "year", "vehicle_age", "km_driven", "fuel",
              "transmission", "owner", "seller_type", "selling_price"]:
        if c in subset.columns:
            display_cols.append(c)

    result = subset[display_cols].head(n).reset_index(drop=True)
    return result


# ══════════════════════════════════════════════════════════════════
# Depreciation Curve
# ══════════════════════════════════════════════════════════════════

def compute_depreciation_curve(
    base_input: Dict[str, Any],
    model,
    preprocessor,
    feature_columns: List[str],
    max_age: int = 20,
) -> List[Dict[str, float]]:
    """
    Predicts price at ages 0..max_age by varying vehicle_age and scaling km_driven.

    km_driven is scaled proportionally to maintain the same km_per_year ratio.
    Returns list of {"age": int, "predicted_price": float}.
    """
    base_age = max(base_input.get("vehicle_age", 5), 0.5)
    base_km = base_input.get("km_driven", 50000)
    km_per_year = base_km / base_age

    curve = []
    for age in range(0, max_age + 1):
        row = dict(base_input)
        row["vehicle_age"] = age
        row["km_driven"] = round(km_per_year * max(age, 0.5))

        # Recompute derived flags and mileage features
        eff_age = max(age, 0.5)
        km = row["km_driven"]
        row["km_per_year"] = round(km / eff_age, 1)
        row["high_mileage_flag"] = 1 if row["km_per_year"] > 20000 else 0
        # km_ratio: use whatever typical_km_per_year was stored in base_input, fallback 10000
        _tkm = float(base_input.get("_typical_km_per_year", 10000.0))
        expected = eff_age * (_tkm if _tkm > 0 else 10000.0)
        row["km_ratio"] = float(np.clip(km / max(expected, 1), 0.05, 10.0))

        # Build input DataFrame with available columns
        try:
            input_df = _build_input_df(row, feature_columns)
            vec = preprocessor.transform(input_df)
            price = float(model.predict(vec)[0])
            curve.append({"age": age, "predicted_price": max(0.0, price)})
        except Exception:
            continue

    return curve


# ══════════════════════════════════════════════════════════════════
# Factor Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════

def compute_factor_sensitivity(
    base_input: Dict[str, Any],
    model,
    preprocessor,
    feature_columns: List[str],
    training_stats: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    For each numeric factor, swaps to median training value and predicts.

    Returns dict: {factor_name: {base_val, swapped_val, price_delta, pct_delta}}.
    """
    base_df = _build_input_df(base_input, feature_columns)
    base_vec = preprocessor.transform(base_df)
    base_price = float(model.predict(base_vec)[0])

    sensitivity = {}
    numeric_factors = ["vehicle_age", "km_driven", "engine_cc", "mileage_val",
                       "max_power_bhp", "owner_tier", "seats", "torque_nm"]

    medians = training_stats.get("medians", {})
    for factor in numeric_factors:
        if factor not in base_input or factor not in medians:
            continue

        base_val = base_input[factor]
        swap_val = medians[factor]

        if abs(float(base_val) - float(swap_val)) < 1e-6:
            continue  # No change — skip

        swapped = dict(base_input)
        swapped[factor] = swap_val

        try:
            swapped_df = _build_input_df(swapped, feature_columns)
            swapped_vec = preprocessor.transform(swapped_df)
            swapped_price = float(model.predict(swapped_vec)[0])
            delta = swapped_price - base_price
            pct_delta = (delta / base_price * 100) if base_price != 0 else 0.0

            sensitivity[factor] = {
                "base_val": base_val,
                "swapped_val": swap_val,
                "price_delta": round(delta, 0),
                "pct_delta": round(pct_delta, 2),
            }
        except Exception:
            continue

    return sensitivity


# ══════════════════════════════════════════════════════════════════
# What-If Analysis
# ══════════════════════════════════════════════════════════════════

def compute_what_if(
    base_input: Dict[str, Any],
    factor_name: str,
    new_value: Any,
    model,
    preprocessor,
    feature_columns: List[str],
) -> Dict[str, Any]:
    """
    Returns the price impact of changing a single factor to new_value.

    Returns {base_price, new_price, delta, pct_delta, factor_name, new_value}.
    """
    base_df = _build_input_df(base_input, feature_columns)
    base_vec = preprocessor.transform(base_df)
    base_price = float(model.predict(base_vec)[0])

    modified = dict(base_input)
    modified[factor_name] = new_value

    # Update high_mileage_flag if km_driven or vehicle_age changed
    if factor_name in ("km_driven", "vehicle_age"):
        age = max(modified.get("vehicle_age", 1), 0.5)
        km = modified.get("km_driven", 0)
        modified["high_mileage_flag"] = 1 if (km / age) > 20000 else 0
        modified["km_per_year"] = round(km / age, 1)
        # km_ratio update
        _tkm = float(modified.get("_typical_km_per_year", modified.get("km_per_year", 10000.0)))
        if _tkm <= 0:
            _tkm = 10000.0
        expected = age * _tkm
        modified["km_ratio"] = float(np.clip(km / max(expected, 1), 0.05, 10.0))

    modified_df = _build_input_df(modified, feature_columns)
    modified_vec = preprocessor.transform(modified_df)
    new_price = float(model.predict(modified_vec)[0])

    delta = new_price - base_price
    pct_delta = (delta / base_price * 100) if base_price != 0 else 0.0

    return {
        "factor_name": factor_name,
        "new_value": new_value,
        "base_price": max(0.0, base_price),
        "new_price": max(0.0, new_price),
        "delta": round(delta, 0),
        "pct_delta": round(pct_delta, 2),
    }


# ══════════════════════════════════════════════════════════════════
# Reliability Assessment
# ══════════════════════════════════════════════════════════════════

def evaluate_prediction_reliability(
    tree_model: DecisionTreeRegressor,
    sample_vector: np.ndarray,
    user_inputs: Dict[str, Any],
    calibration_residuals: Optional[np.ndarray] = None,
    known_brands: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Computes an honest reliability assessment for a tree prediction.

    1. Leaf support: Number of training samples in the reached leaf.
    2. Empirical error interval: 85th-percentile absolute residual.
    3. OOD checks: outlier mileage, age, engine displacement, unknown brand.
    """
    if sample_vector.ndim == 1:
        sample_vector = sample_vector.reshape(1, -1)

    leaf_id = int(tree_model.apply(sample_vector)[0])
    leaf_samples = int(tree_model.tree_.n_node_samples[leaf_id])

    if leaf_samples >= 20:
        leaf_rating = "High Support"
        leaf_color = "green"
    elif leaf_samples >= 7:
        leaf_rating = "Moderate Support"
        leaf_color = "orange"
    else:
        leaf_rating = "Sparse Support"
        leaf_color = "red"

    # OOD checks
    flags = []
    brand = user_inputs.get("brand", "Unknown")
    if known_brands and brand not in known_brands:
        flags.append(f"Brand '{brand}' was not present in the training distribution.")

    km = user_inputs.get("km_driven", 0)
    if km > 250000:
        flags.append(f"Extreme odometer mileage ({km:,} km) exceeds typical market ranges.")
    elif km < 500:
        flags.append(f"Very low mileage ({km} km) suggests near-new or demonstration vehicle.")

    age = user_inputs.get("vehicle_age", 0)
    if age > 18:
        flags.append(f"Vehicle age ({age} years) is at the tail end of the market.")

    engine = user_inputs.get("engine_cc", 1200)
    if engine > 4500:
        flags.append(f"Engine displacement ({engine} CC) represents an exotic or commercial vehicle.")

    # Empirical Prediction Interval
    if calibration_residuals is not None and len(calibration_residuals) > 0:
        margin = float(np.percentile(np.abs(calibration_residuals), 85))
    else:
        margin = 65000.0

    return {
        "leaf_id": leaf_id,
        "leaf_samples": leaf_samples,
        "support_level": leaf_rating,
        "support_color": leaf_color,
        "margin_of_error": margin,
        "ood_flags": flags,
        "is_out_of_distribution": len(flags) > 0,
        "explanation": (
            f"Reached Leaf Node #{leaf_id} containing {leaf_samples} training vehicles ({leaf_rating}). "
            f"Empirical 85% residual margin is ±{format_inr(margin)}."
        ),
    }


# ══════════════════════════════════════════════════════════════════
# Market Positioning
# ══════════════════════════════════════════════════════════════════

def compute_market_positioning(
    estimated_price: float,
    asking_price: Optional[float],
    tolerance_pct: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    Compares seller asking price against model estimate with a tolerance corridor.
    """
    if asking_price is None or asking_price <= 0:
        return None

    diff_abs = asking_price - estimated_price
    diff_pct = (diff_abs / estimated_price) * 100.0
    tol_bound = estimated_price * (tolerance_pct / 100.0)

    lower_bound = estimated_price - tol_bound
    upper_bound = estimated_price + tol_bound

    if asking_price < lower_bound:
        positioning = "Potentially Underpriced (Favorable Buyer Deal)"
        status_color = "green"
        recommendation = (
            "The vehicle is listed below the model estimate, which may indicate a strong deal "
            "or require a physical condition inspection before purchase."
        )
    elif asking_price > upper_bound:
        positioning = "Potentially Overpriced (Above Market Median)"
        status_color = "red"
        recommendation = (
            "The vehicle is listed above typical market value. "
            "Room for price negotiation is recommended."
        )
    else:
        positioning = "Within Estimated Fair Market Range"
        status_color = "blue"
        recommendation = (
            "The listing price closely aligns with model expectations "
            "based on age, mileage, and specifications."
        )

    return {
        "estimated_price": estimated_price,
        "asking_price": asking_price,
        "diff_abs": diff_abs,
        "diff_pct": diff_pct,
        "tolerance_pct": tolerance_pct,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "positioning": positioning,
        "status_color": status_color,
        "recommendation": recommendation,
    }


# ══════════════════════════════════════════════════════════════════
# Internal Helper
# ══════════════════════════════════════════════════════════════════

def _build_input_df(user_input: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """
    Constructs a single-row DataFrame aligned to expected feature columns.
    Missing columns are filled with np.nan (imputed by the preprocessor).
    """
    row = {}
    for col in feature_columns:
        row[col] = user_input.get(col, np.nan)
    return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════
# Mileage Adjustment Layer (Isotonic Post-Processing)
# ══════════════════════════════════════════════════════════════════

# Default clipping bounds for the adjustment factor
_ADJ_LOWER = 0.80   # max 20% downward correction
_ADJ_UPPER = 1.15   # max 15% upward correction


def fit_mileage_adjuster(
    km_ratio: np.ndarray,
    actual_prices: np.ndarray,
    cart_prices: np.ndarray,
    adj_lower: float = _ADJ_LOWER,
    adj_upper: float = _ADJ_UPPER,
) -> IsotonicRegression:
    """
    Fits an IsotonicRegression that maps km_ratio → residual_ratio.

    residual_ratio = actual / cart_pred for each validation sample.
    The isotonic fit is monotone decreasing (increasing km_ratio → smaller ratio,
    i.e. CART base over-estimates high-km cars and under-estimates low-km ones).

    Args:
        km_ratio:      1-D array, km_driven / (vehicle_age * typical_km_per_year)
        actual_prices: 1-D array of ground-truth selling prices (validation set)
        cart_prices:   1-D array of CART predictions for the same rows
        adj_lower:     Minimum adjustment factor (clips fitted values)
        adj_upper:     Maximum adjustment factor

    Returns:
        Fitted IsotonicRegression (increasing=False).
    """
    # Avoid division by zero
    safe_cart = np.where(cart_prices > 0, cart_prices, 1.0)
    residual_ratio = actual_prices / safe_cart
    # Clip to sensible range before fitting to avoid extreme training artefacts
    residual_ratio = np.clip(residual_ratio, adj_lower, adj_upper)

    adjuster = IsotonicRegression(increasing=False, out_of_bounds="clip")
    adjuster.fit(km_ratio, residual_ratio)
    return adjuster


def apply_mileage_adjustment(
    cart_price: float,
    km_ratio: float,
    adjuster: Optional[IsotonicRegression],
    adj_lower: float = _ADJ_LOWER,
    adj_upper: float = _ADJ_UPPER,
    enabled: bool = True,
) -> Dict[str, float]:
    """
    Applies the isotonic mileage adjustment to a CART base prediction.

    Returns:
        {
          "cart_base":          CART base price (₹),
          "adjustment_factor":  isotonic factor (clipped to [adj_lower, adj_upper]),
          "final_price":        cart_base * adjustment_factor,
          "enabled":            whether adjustment was applied
        }
    """
    if not enabled or adjuster is None:
        return {
            "cart_base": cart_price,
            "adjustment_factor": 1.0,
            "final_price": cart_price,
            "enabled": False,
        }

    raw_factor = float(adjuster.predict(np.array([km_ratio]))[0])
    factor = float(np.clip(raw_factor, adj_lower, adj_upper))
    final = max(0.0, cart_price * factor)

    return {
        "cart_base": cart_price,
        "adjustment_factor": round(factor, 4),
        "final_price": round(final, 2),
        "enabled": True,
    }


def compute_mileage_impact_curve(
    base_input: Dict[str, Any],
    model,
    preprocessor,
    feature_columns: List[str],
    adjuster: Optional[IsotonicRegression],
    typical_km_per_year: float,
    n_points: int = 20,
    adj_lower: float = _ADJ_LOWER,
    adj_upper: float = _ADJ_UPPER,
) -> List[Dict[str, Any]]:
    """
    Varies km_driven from 10% to 300% of the entered value and returns
    CART base price + adjusted price at each point.

    Returns list of dicts:
        {km, cart_price, adjusted_price, km_ratio, pct_vs_base}
    where pct_vs_base is % change relative to km_ratio=1 reference point.
    """
    base_km = max(float(base_input.get("km_driven", 50000)), 500.0)
    vehicle_age = max(float(base_input.get("vehicle_age", 5)), 0.5)

    lo = max(500.0, base_km * 0.10)
    hi = base_km * 3.0
    km_values = np.linspace(lo, hi, n_points)

    results = []
    ref_cart = None

    for km in km_values:
        row = dict(base_input)
        row["km_driven"] = float(km)
        # Update derived mileage features consistently
        row["km_per_year"] = round(km / vehicle_age, 1)
        row["high_mileage_flag"] = 1 if row["km_per_year"] > 20000 else 0
        expected_km = vehicle_age * (typical_km_per_year if typical_km_per_year > 0 else 10000.0)
        km_r = float(np.clip(km / max(expected_km, 1), 0.05, 10.0))
        row["km_ratio"] = round(km_r, 4)

        try:
            input_df = _build_input_df(row, feature_columns)
            vec = preprocessor.transform(input_df)
            cart_p = max(0.0, float(model.predict(vec)[0]))
            adj = apply_mileage_adjustment(cart_p, km_r, adjuster,
                                           adj_lower, adj_upper, enabled=adjuster is not None)
            adj_p = adj["final_price"]

            if ref_cart is None:
                ref_cart = cart_p if cart_p > 0 else 1.0

            results.append({
                "km": int(round(km)),
                "km_ratio": round(km_r, 3),
                "cart_price": cart_p,
                "adjusted_price": adj_p,
                "pct_vs_base": round((adj_p / ref_cart - 1) * 100, 2),
                "adjustment_factor": adj["adjustment_factor"],
            })
        except Exception:
            continue

    return results
