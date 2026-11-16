"""
Unit tests for Data Validation and Quality Assurance.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_validation import extract_numeric_value, validate_raw_schema, perform_data_validation


def test_extract_numeric_value():
    assert extract_numeric_value("1248 CC") == 1248.0
    assert extract_numeric_value("23.4 kmpl") == 23.4
    assert extract_numeric_value("74 bhp") == 74.0
    assert extract_numeric_value(1500) == 1500.0
    assert extract_numeric_value(None) is not None and np.isnan(extract_numeric_value(None))


def test_validate_raw_schema():
    df_valid = pd.DataFrame({
        "name": ["Maruti Swift"],
        "year": [2018],
        "selling_price": [450000],
        "km_driven": [35000]
    })
    is_valid, missing = validate_raw_schema(df_valid)
    assert is_valid is True
    assert len(missing) == 0

    df_invalid = pd.DataFrame({
        "random_col": [1]
    })
    is_valid, missing = validate_raw_schema(df_invalid)
    assert is_valid is False
    assert len(missing) > 0


def test_perform_data_validation():
    # Construct synthetic dirty data for testing edge cases
    dirty_data = pd.DataFrame({
        "name": ["Maruti Swift", "Maruti Swift", "Hyundai i20", "Invalid Price Car", "Negative Year Car"],
        "year": [2018, 2018, 2019, 2015, 1970],
        "selling_price": ["₹ 4,50,000", "₹ 4,50,000", "5,50,000", -500, "15,00,000"],
        "km_driven": ["35,000 km", "35,000 km", "40,000", "20,000", "50,000"],
        "fuel": ["Petrol", "Petrol", "Diesel", "Diesel", "Petrol"],
        "transmission": ["Manual", "Manual", "Automatic", "Manual", "Manual"],
        "engine": ["1197 CC", "1197 CC", "1396 CC", "1197 CC", "1197 CC"]
    })

    cleaned, audit = perform_data_validation(dirty_data, reference_year=2024)
    assert audit["initial_record_count"] == 5
    # Duplicate removed, negative price removed, negative year removed
    assert len(cleaned) == 2
    assert audit["removed_records"]["duplicate_rows"] == 1
    assert audit["removed_records"]["invalid_prices"] >= 1
    assert all(cleaned["selling_price"] > 25000)
    assert all(cleaned["year"] >= 1985)
