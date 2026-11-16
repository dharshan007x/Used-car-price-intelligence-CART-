"""
Unit tests for EDA and Feature Engineering modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import extract_brand_and_model, engineer_features
from src.eda import compute_dataset_overview, compute_statistical_summary, generate_dynamic_eda_insights
from src.evaluation import compute_comprehensive_metrics, compute_segment_metrics


def test_extract_brand_and_model():
    brand, model = extract_brand_and_model("Maruti Swift Dzire VDI")
    assert brand == "Maruti"
    assert model == "Swift Dzire"  # multi-word model correctly parsed

    brand, model = extract_brand_and_model("Land Rover Range Rover Evoque")
    assert brand == "Land Rover"
    assert model == "Range Rover"  # multi-word model correctly parsed

    brand, model = extract_brand_and_model("Hyundai")
    assert brand == "Hyundai"
    assert model == "Standard"

    brand, model = extract_brand_and_model("")
    assert brand == "Unknown"


def test_engineer_features():
    df = pd.DataFrame({
        "name": ["Maruti Swift Dzire", "Hyundai i20 Magna"],
        "year": [2020, 2018],
        "km_driven": [40000, 60000],
        "selling_price": [550000, 480000],
        "engine_cc": [1197, 1396]
    })
    df_eng = engineer_features(df, reference_year=2024)
    assert "brand" in df_eng.columns
    assert "model_name" in df_eng.columns
    assert "vehicle_age" in df_eng.columns
    assert "km_per_year" in df_eng.columns
    assert df_eng["vehicle_age"].iloc[0] == 4
    assert df_eng["vehicle_age"].iloc[1] == 6


def test_eda_computations():
    df = pd.DataFrame({
        "brand": ["Maruti", "Maruti", "Hyundai", "Honda"],
        "selling_price": [400000, 500000, 600000, 700000],
        "vehicle_age": [6, 4, 3, 2],
        "km_driven": [60000, 40000, 30000, 20000],
        "fuel": ["Petrol", "Diesel", "Petrol", "Petrol"],
        "transmission": ["Manual", "Manual", "Automatic", "Automatic"],
        "engine_cc": [1197, 1248, 1197, 1498]
    })

    overview = compute_dataset_overview(df)
    assert overview["total_records"] == 4
    assert overview["target_stats"]["mean"] == 550000.0

    stats = compute_statistical_summary(df)
    assert "Mean" in stats.columns
    assert "Median" in stats.columns

    insights = generate_dynamic_eda_insights(df)
    assert isinstance(insights, list)


def test_evaluation_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    metrics = compute_comprehensive_metrics(y_true, y_pred)
    assert metrics["mae"] == 10.0
    assert metrics["rmse"] == 10.0
    assert metrics["r2"] > 0.95
    assert metrics["mape_pct"] > 0

    # Test price segments
    y_t_real = np.array([200000, 500000, 1200000, 2500000])
    y_p_real = np.array([210000, 490000, 1150000, 2600000])
    seg_df = compute_segment_metrics(y_t_real, y_p_real)
    assert len(seg_df) == 4
