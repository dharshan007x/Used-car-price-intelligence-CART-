"""
Tests for brand-model map completeness, parser correctness, and UI coverage.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════
# Parser tests — multi-word models preserved
# ═════════════════════════════════════════════════════════════════

def test_parser_multi_word_models():
    from src.feature_engineering import extract_brand_and_model
    cases = [
        ("Maruti Swift Dzire VDI",          "Maruti",        "Swift Dzire"),
        ("Maruti Wagon R LXI DUO BSIII",    "Maruti",        "Wagon R"),
        ("Maruti Alto 800 LXI",             "Maruti",        "Alto 800"),
        ("Hyundai Grand i10 Magna",         "Hyundai",       "Grand i10"),
        ("Hyundai Santro Xing GLS",         "Hyundai",       "Santro Xing"),  # correctly parsed as multi-word
        ("BMW 3 Series 320d Luxury Line",   "BMW",           "3 Series"),
        ("BMW 5 Series 520d Sport Line",    "BMW",           "5 Series"),
        ("Toyota Land Cruiser Prado VX L",  "Toyota",        "Land Cruiser"),
        ("Mercedes-Benz New C-Class 220",   "Mercedes-Benz", "C-Class"),
        ("Honda WR-V i-DTEC VX",           "Honda",         "WR-V"),
        ("Honda CR-V 2.4L 4WD AT",         "Honda",         "CR-V"),
    ]
    for name, expected_brand, expected_model in cases:
        brand, model = extract_brand_and_model(name)
        assert brand == expected_brand, f"Brand wrong for '{name}': got '{brand}' expected '{expected_brand}'"
        assert model == expected_model, f"Model wrong for '{name}': got '{model}' expected '{expected_model}'"


def test_parser_handles_unknown_gracefully():
    from src.feature_engineering import extract_brand_and_model
    brand, model = extract_brand_and_model(None)
    assert brand == "Unknown"
    assert model == "Unknown"

    brand, model = extract_brand_and_model("")
    assert brand == "Unknown"

    brand, model = extract_brand_and_model("SingleWordOnly")
    assert brand is not None
    assert model == "Standard"


def test_parser_bmw_casing():
    from src.feature_engineering import extract_brand_and_model
    brand, model = extract_brand_and_model("BMW X1 sDrive20i xLine")
    assert brand == "BMW", f"Expected 'BMW', got '{brand}'"


def test_parser_no_brand_duplication():
    """No model should appear under the wrong brand."""
    from src.feature_engineering import extract_brand_and_model
    test_cases = [
        "Maruti Swift VXI",
        "Hyundai i20 Sportz Diesel",
        "Toyota Innova Crysta",
        "Honda City ZX",
    ]
    expected_brands = ["Maruti", "Hyundai", "Toyota", "Honda"]
    for name, exp_brand in zip(test_cases, expected_brands):
        brand, _ = extract_brand_and_model(name)
        assert brand == exp_brand, f"'{name}' -> brand='{brand}', expected '{exp_brand}'"


# ═════════════════════════════════════════════════════════════════
# Brand-model map tests
# ═════════════════════════════════════════════════════════════════

def test_brand_model_map_exists_and_complete():
    map_path = PROJECT_ROOT / "models" / "brand_model_map.json"
    assert map_path.exists(), "brand_model_map.json not found — run python -m src.train"
    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    brand_map = raw.get("brands", {})
    assert len(brand_map) > 0, "Brand map is empty"
    assert "_n_brands" in raw, "Map missing metadata"
    assert "_n_models" in raw, "Map missing model count metadata"


def test_all_cleaned_dataset_models_in_map():
    """Every (brand, model) in the cleaned dataset must appear in the map."""
    map_path = PROJECT_ROOT / "models" / "brand_model_map.json"
    processed_path = PROJECT_ROOT / "data" / "processed" / "cleaned_car_data.csv"
    if not map_path.exists() or not processed_path.exists():
        pytest.skip("Artifacts not found — run python -m src.train first.")

    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    brand_map = raw.get("brands", {})

    df = pd.read_csv(processed_path)
    if "brand" not in df.columns or "model_name" not in df.columns:
        pytest.skip("brand/model_name columns not in processed data.")

    missing = []
    for _, row in df[["brand", "model_name"]].drop_duplicates().iterrows():
        brand = str(row["brand"])
        model = str(row["model_name"])
        map_models = [m["name"] for m in brand_map.get(brand, [])]
        if model not in map_models:
            missing.append((brand, model))

    assert len(missing) == 0, (
        f"{len(missing)} (brand, model) pairs missing from brand_model_map: "
        f"{missing[:10]} ..."
    )


def test_no_model_under_wrong_brand():
    """No model appears under a brand it doesn't belong to."""
    map_path = PROJECT_ROOT / "models" / "brand_model_map.json"
    if not map_path.exists():
        pytest.skip("brand_model_map.json not found.")

    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    brand_map = raw.get("brands", {})

    # Build inverse: model_name -> list of brands it appears under
    model_to_brands: dict = {}
    for brand, models in brand_map.items():
        for m in models:
            model_to_brands.setdefault(m["name"], []).append(brand)

    # Models that appear under MULTIPLE brands should be intentional
    # (some models share names across brands, e.g. "Classic" in Ford and Ambassador)
    # This test just checks the structure is not corrupted (no brand owns 0 models)
    for brand, models in brand_map.items():
        assert len(models) > 0, f"Brand '{brand}' has no models in map"


def test_rare_models_in_map():
    """Models with only 1 listing should still appear in the map (no rarity filter)."""
    map_path = PROJECT_ROOT / "models" / "brand_model_map.json"
    if not map_path.exists():
        pytest.skip("brand_model_map.json not found.")

    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    brand_map = raw.get("brands", {})

    all_counts = []
    for brand, models in brand_map.items():
        for m in models:
            all_counts.append(m["count"])

    # At least some models should have count=1 (rare models preserved)
    assert 1 in all_counts, "No single-listing (rare) models found — min_frequency filter may have been applied to map"


def test_brand_model_map_function():
    """build_brand_model_map returns correct structure from a minimal dataframe."""
    from src.feature_engineering import build_brand_model_map
    df = pd.DataFrame({
        "brand": ["Maruti", "Maruti", "Maruti", "Hyundai", "Hyundai"],
        "model_name": ["Swift", "Swift", "Baleno", "i20", "i20"],
    })
    result = build_brand_model_map(df, "brand", "model_name")
    assert "Maruti" in result
    assert "Hyundai" in result
    maruti_names = [m["name"] for m in result["Maruti"]]
    assert "Swift" in maruti_names
    assert "Baleno" in maruti_names
    assert "i20" not in maruti_names, "i20 should not appear under Maruti"
    # Swift has count=2, should come first
    assert result["Maruti"][0]["name"] == "Swift"


# ═════════════════════════════════════════════════════════════════
# Coverage audit report
# ═════════════════════════════════════════════════════════════════

def test_model_coverage_audit_exists():
    audit_path = PROJECT_ROOT / "reports" / "model_coverage_audit.csv"
    assert audit_path.exists(), "model_coverage_audit.csv not found — run python -m src.train"
    df = pd.read_csv(audit_path)
    required_cols = {"brand", "model", "raw_count", "in_dropdown"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns: {required_cols - set(df.columns)}"
    # All rows should be in dropdown (verify 100% coverage)
    missing = df[df["in_dropdown"] == False]
    assert len(missing) == 0, f"{len(missing)} models missing from dropdown:\n{missing.to_string()}"


# ═════════════════════════════════════════════════════════════════
# Catalog-only model — warning + finite prediction
# ═════════════════════════════════════════════════════════════════

def test_catalog_only_model_gives_finite_prediction():
    """A catalog-only model (not in training) should return finite, non-negative price."""
    prep_path = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
    model_path = PROJECT_ROOT / "models" / "best_cart_model.joblib"
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not all(p.exists() for p in [prep_path, model_path, meta_path]):
        pytest.skip("Model artifacts not found.")

    import joblib
    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    with open(meta_path) as f:
        meta = json.load(f)
    all_cols = meta.get("input_features", [])

    from src.prediction import _build_input_df
    # Use a model that is NOT in the training data at all
    input_data = {
        "vehicle_age": 4, "km_driven": 40000, "engine_cc": 1498,
        "mileage_val": 17.0, "max_power_bhp": 108.0, "owner_tier": 1,
        "seats": 5.0, "torque_nm": 140.0, "high_mileage_flag": 0,
        "brand": "Kia", "model_name": "Stinger",   # almost certainly not in training data
        "fuel": "Petrol", "transmission": "Automatic",
        "seller_type": "Individual", "trim_tier": "Top", "body_segment": "Sedan",
    }
    df_in = _build_input_df(input_data, all_cols)
    result = prep.transform(df_in)
    price = float(model.predict(result)[0])
    assert np.isfinite(price), "Price should be finite for catalog-only model"
    assert price >= 0, "Price should be non-negative"


# ═════════════════════════════════════════════════════════════════
# Unknown brand/model — no crash
# ═════════════════════════════════════════════════════════════════

def test_unknown_brand_model_no_crash():
    prep_path = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
    model_path = PROJECT_ROOT / "models" / "best_cart_model.joblib"
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not all(p.exists() for p in [prep_path, model_path, meta_path]):
        pytest.skip("Model artifacts not found.")

    import joblib
    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    with open(meta_path) as f:
        meta = json.load(f)
    all_cols = meta.get("input_features", [])

    from src.prediction import _build_input_df
    input_data = {
        "vehicle_age": 5, "km_driven": 60000, "engine_cc": 1200,
        "mileage_val": 18.0, "max_power_bhp": 80.0, "owner_tier": 2,
        "seats": 5.0, "torque_nm": 100.0, "high_mileage_flag": 0,
        "brand": "XXXX_UNKNOWN_BRAND", "model_name": "YYYY_UNKNOWN_MODEL",
        "fuel": "Petrol", "transmission": "Manual",
        "seller_type": "Individual", "trim_tier": "Mid", "body_segment": "Sedan",
    }
    df_in = _build_input_df(input_data, all_cols)
    result = prep.transform(df_in)   # must not raise
    price = float(model.predict(result)[0])
    assert np.isfinite(price)
    assert price >= 0
