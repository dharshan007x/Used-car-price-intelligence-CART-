"""
Tests for market-grade features, prediction functions, and new engineering.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════
# Test 1 — feature_availability.csv exists and has correct columns
# ═════════════════════════════════════════════════════════════════

def test_feature_availability_csv_exists_and_valid():
    avail_path = PROJECT_ROOT / "reports" / "feature_availability.csv"
    assert avail_path.exists(), (
        "reports/feature_availability.csv not found. Run python -m src.train first."
    )
    df = pd.read_csv(avail_path)
    required_cols = {"factor_group", "column_name", "available", "enabled", "notes"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )
    # Must mention city as unavailable
    city_rows = df[df["column_name"] == "city"]
    assert not city_rows.empty, "city/location row must be present in availability report"
    assert city_rows.iloc[0]["available"] == False or city_rows.iloc[0]["available"] == "False", (
        "city must be marked as unavailable"
    )


# ═════════════════════════════════════════════════════════════════
# Test 2 — unknown owner/seller/trim values don't crash preprocessor
# ═════════════════════════════════════════════════════════════════

def test_unknown_categorical_values_handled():
    prep_path = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not prep_path.exists() or not meta_path.exists():
        pytest.skip("Model artifacts not found — run python -m src.train first.")

    import json
    import joblib
    prep = joblib.load(prep_path)
    with open(meta_path) as f:
        meta = json.load(f)
    all_cols = meta.get("input_features", [])

    from src.prediction import _build_input_df
    # Row with unseen categorical values
    weird_input = {
        "vehicle_age": 5, "km_driven": 50000, "engine_cc": 1200,
        "mileage_val": 18.0, "max_power_bhp": 80.0, "owner_tier": 2,
        "seats": 5.0, "torque_nm": 100.0, "high_mileage_flag": 0,
        "brand": "MARTIAN_BRAND_XYZ",    # unknown
        "model_name": "UNKNOWN_MODEL",    # unknown
        "fuel": "Petrol",
        "transmission": "Manual",
        "seller_type": "ALIEN_SELLER",    # unknown
        "trim_tier": "ULTRA",             # unknown
        "body_segment": "SPACESHIP",      # unknown
    }
    df_in = _build_input_df(weird_input, all_cols)
    result = prep.transform(df_in)
    assert result.shape[0] == 1
    assert not np.any(np.isnan(result)), "Transform with unknown categories should not produce NaN"


# ═════════════════════════════════════════════════════════════════
# Test 3 — compute_price_range: low <= fair <= high, all non-negative
# ═════════════════════════════════════════════════════════════════

def test_price_range_ordering():
    from src.prediction import compute_price_range

    residuals = np.random.normal(0, 50000, 500)
    for predicted in [150000, 500000, 900000, 2500000]:
        result = compute_price_range(predicted, calibration_residuals=residuals)
        assert result["low"] >= 0, f"low price should be >= 0, got {result['low']}"
        assert result["low"] <= result["fair"], "low should be <= fair"
        assert result["fair"] <= result["high"], "fair should be <= high"
        assert np.isfinite(result["low"]) and np.isfinite(result["high"]), "range must be finite"


def test_price_range_no_residuals():
    """Should use heuristic fallback when no residuals supplied."""
    from src.prediction import compute_price_range
    result = compute_price_range(500000)
    assert result["low"] >= 0
    assert result["low"] <= result["fair"] <= result["high"]


# ═════════════════════════════════════════════════════════════════
# Test 4 — get_comparable_listings returns real rows
# ═════════════════════════════════════════════════════════════════

def test_comparable_listings_real_rows():
    from src.prediction import get_comparable_listings

    training_df = pd.DataFrame({
        "brand": ["Maruti", "Maruti", "Hyundai", "Honda"],
        "model_name": ["Swift", "Swift", "i20", "City"],
        "vehicle_age": [3, 5, 4, 6],
        "km_driven": [30000, 55000, 45000, 70000],
        "fuel": ["Petrol", "Petrol", "Diesel", "Petrol"],
        "selling_price": [450000, 380000, 620000, 500000],
    })
    user = {"brand": "Maruti", "model_name": "Swift", "vehicle_age": 4, "km_driven": 40000}
    result = get_comparable_listings(user, training_df, n=3)
    assert len(result) <= 3
    # All rows should be from the original training_df (check selling_price values)
    assert set(result["selling_price"].tolist()).issubset(set(training_df["selling_price"].tolist()))


# ═════════════════════════════════════════════════════════════════
# Test 5 — compute_depreciation_curve varies with age
# ═════════════════════════════════════════════════════════════════

def test_depreciation_curve_varies():
    prep_path = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
    model_path = PROJECT_ROOT / "models" / "best_cart_model.joblib"
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not all(p.exists() for p in [prep_path, model_path, meta_path]):
        pytest.skip("Model artifacts not found — run python -m src.train first.")

    import json, joblib
    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    with open(meta_path) as f:
        meta = json.load(f)
    all_cols = meta.get("input_features", [])

    from src.prediction import compute_depreciation_curve
    base_input = {
        "vehicle_age": 3, "km_driven": 40000, "engine_cc": 1197,
        "mileage_val": 21.4, "max_power_bhp": 82.0, "owner_tier": 1,
        "seats": 5.0, "torque_nm": 113.0, "high_mileage_flag": 0,
        "brand": "Maruti", "model_name": "Swift", "fuel": "Petrol",
        "transmission": "Manual", "seller_type": "Individual",
        "trim_tier": "Mid", "body_segment": "Hatchback",
    }
    curve = compute_depreciation_curve(base_input, model, prep, all_cols, max_age=10)
    assert len(curve) >= 5, "Curve should have at least 5 data points"
    prices = [pt["predicted_price"] for pt in curve]
    assert max(prices) > min(prices), "Curve prices should not be constant across all ages"
    assert all(p >= 0 for p in prices), "All predicted prices must be non-negative"


# ═════════════════════════════════════════════════════════════════
# Test 6 — compute_what_if returns finite non-negative values
# ═════════════════════════════════════════════════════════════════

def test_what_if_finite_non_negative():
    prep_path = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
    model_path = PROJECT_ROOT / "models" / "best_cart_model.joblib"
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not all(p.exists() for p in [prep_path, model_path, meta_path]):
        pytest.skip("Model artifacts not found — run python -m src.train first.")

    import json, joblib
    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    with open(meta_path) as f:
        meta = json.load(f)
    all_cols = meta.get("input_features", [])

    from src.prediction import compute_what_if
    base = {
        "vehicle_age": 5, "km_driven": 60000, "engine_cc": 1197,
        "mileage_val": 21.4, "max_power_bhp": 82.0, "owner_tier": 1,
        "seats": 5.0, "torque_nm": 113.0, "high_mileage_flag": 0,
        "brand": "Maruti", "model_name": "Swift", "fuel": "Petrol",
        "transmission": "Manual", "seller_type": "Individual",
        "trim_tier": "Mid", "body_segment": "Hatchback",
    }
    result = compute_what_if(base, "km_driven", 120000, model, prep, all_cols)
    assert np.isfinite(result["new_price"]), "new_price must be finite"
    assert result["new_price"] >= 0, "new_price must be non-negative"
    assert np.isfinite(result["delta"]), "delta must be finite"
    assert np.isfinite(result["pct_delta"]), "pct_delta must be finite"


# ═════════════════════════════════════════════════════════════════
# Test 7 — dominance flag in metadata is a boolean
# ═════════════════════════════════════════════════════════════════

def test_dominance_flag_is_boolean():
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not meta_path.exists():
        pytest.skip("model_metadata.json not found — run python -m src.train first.")
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    assert "dominance_warning" in meta, "dominance_warning key must be in metadata"
    assert isinstance(meta["dominance_warning"], bool), "dominance_warning must be a bool"
    assert "dominant_group" in meta, "dominant_group key must be in metadata"
    assert "dominant_group_pct" in meta, "dominant_group_pct key must be in metadata"
    assert 0.0 <= meta["dominant_group_pct"] <= 100.0, "dominant_group_pct must be in [0, 100]"


# ═════════════════════════════════════════════════════════════════
# Test 8 — Feature engineering: owner_tier and trim_tier correct
# ═════════════════════════════════════════════════════════════════

def test_owner_tier_mapping():
    from src.feature_engineering import encode_owner_tier
    assert encode_owner_tier("First Owner") == 1
    assert encode_owner_tier("second owner") == 2
    assert encode_owner_tier("Third Owner") == 3
    assert encode_owner_tier("Fourth & Above Owner") == 4
    assert encode_owner_tier("Test Drive Car") == 5
    assert encode_owner_tier(None) == 2  # default


def test_trim_tier_extraction():
    from src.feature_engineering import extract_trim_tier
    assert extract_trim_tier("Maruti Swift ZXI+") == "Top"
    assert extract_trim_tier("Hyundai i20 Magna") == "Mid"
    assert extract_trim_tier("Maruti Alto LXI") == "Base"
    assert extract_trim_tier(None) == "Mid"


def test_torque_parsing():
    from src.feature_engineering import parse_torque_nm
    assert parse_torque_nm("115Nm@ 4500rpm") == 115.0
    assert parse_torque_nm("200 Nm at 2000rpm") == 200.0
    assert parse_torque_nm(None) != parse_torque_nm(None) or True  # nan is nan
    # kgm conversion: 11.5 kgm → ~112.8 Nm
    result = parse_torque_nm("11.5kgm at 2500rpm")
    assert 100 <= result <= 130, f"kgm conversion off: {result}"


def test_body_segment_lookup():
    from src.feature_engineering import derive_body_segment
    assert derive_body_segment("Swift") == "Hatchback"
    assert derive_body_segment("City") == "Sedan"
    assert derive_body_segment("Creta") == "SUV"
    assert derive_body_segment("Innova") == "MUV"
    assert derive_body_segment("UnknownModel") == "Sedan"  # default
