"""
Unit tests for Preprocessing Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import build_preprocessor, get_feature_names_from_preprocessor


def test_preprocessing_pipeline_fit_transform():
    df_train = pd.DataFrame({
        "vehicle_age": [3, 5, 8],
        "km_driven": [30000, 50000, 90000],
        "engine_cc": [1197, 1498, 1995],
        "mileage_val": [21.4, 18.5, 14.2],
        "max_power_bhp": [82.0, 100.0, 140.0],
        "brand": ["Maruti", "Honda", "BMW"],
        "fuel": ["Petrol", "Diesel", "Diesel"],
        "transmission": ["Manual", "Manual", "Automatic"]
    })

    num_cols = ["vehicle_age", "km_driven", "engine_cc", "mileage_val", "max_power_bhp"]
    cat_cols = ["brand", "fuel", "transmission"]

    preprocessor = build_preprocessor(num_features=num_cols, cat_features=cat_cols)
    transformed = preprocessor.fit_transform(df_train)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > len(num_cols) # due to one-hot encoding

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    assert len(feature_names) == transformed.shape[1]


def test_preprocessing_unknown_categories():
    df_train = pd.DataFrame({
        "vehicle_age": [3, 5],
        "km_driven": [30000, 50000],
        "engine_cc": [1200, 1500],
        "brand": ["Maruti", "Honda"],
        "fuel": ["Petrol", "Diesel"],
        "transmission": ["Manual", "Automatic"]
    })
    preprocessor = build_preprocessor(
        num_features=["vehicle_age", "km_driven", "engine_cc"],
        cat_features=["brand", "fuel", "transmission"]
    )
    preprocessor.fit(df_train)

    # Test sample with completely unseen brand ("Ferrari") and unseen fuel ("Electric")
    df_unseen = pd.DataFrame({
        "vehicle_age": [1],
        "km_driven": [1000],
        "engine_cc": [3900],
        "brand": ["Ferrari"],
        "fuel": ["Electric"],
        "transmission": ["Automatic"]
    })

    # Should not raise ValueError (handle_unknown='ignore')
    trans_unseen = preprocessor.transform(df_unseen)
    assert trans_unseen.shape[0] == 1
    assert not np.isnan(trans_unseen).any()
