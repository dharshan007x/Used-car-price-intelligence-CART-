"""
Data Validation and Quality Assurance Module for Used Car Price Intelligence.

Performs schema verification, anomaly detection, statistical boundaries validation,
and generates comprehensive data quality audit summaries.
"""

from typing import Dict, Any, Tuple
import re
import numpy as np
import pandas as pd


MANDATORY_RAW_COLUMNS = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "transmission",
    "engine"
]

REQUIRED_FINAL_FEATURES = [
    "brand",
    "vehicle_age",
    "km_driven",
    "fuel",
    "engine_cc",
    "transmission",
    "selling_price"
]


def extract_numeric_value(val: Any) -> float:
    """Extracts first float/int value from strings like '1248 CC', '23.4 kmpl', '74 bhp', '₹ 4,50,000'."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    cleaned_str = str(val).replace(",", "").replace(" ", "")
    match = re.search(r"(\d+(\.\d+)?)", cleaned_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def validate_raw_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """Checks whether the dataframe has minimum required raw columns."""
    cols = [c.lower().strip() for c in df.columns]
    missing = []
    
    # Check for direct or fuzzy matches
    for req in ["selling_price", "year", "km_driven"]:
        if not any(req in c for c in cols):
            missing.append(req)
            
    # Check for name/brand
    if not any(c in cols for c in ["name", "brand", "car_name"]):
        missing.append("name/brand")
        
    return len(missing) == 0, missing


def perform_data_validation(df: pd.DataFrame, reference_year: int = 2024) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates, parses dirty strings, sanitizes bounds, and generates a data quality report.
    
    Returns:
        cleaned_df: Cleaned dataframe ready for feature engineering/EDA
        audit_report: Comprehensive summary of validation and changes made
    """
    initial_records = len(df)
    audit: Dict[str, Any] = {
        "initial_record_count": initial_records,
        "initial_columns": list(df.columns),
        "missing_values_per_col": df.isnull().sum().to_dict(),
        "duplicate_records_detected": int(df.duplicated().sum()),
        "removed_records": {}
    }

    cleaned = df.copy()

    # Column name normalization
    cleaned.columns = [c.strip().lower().replace(" ", "_") for c in cleaned.columns]

    # Clean duplicates - CarDekho lists identical entries from multiple dealer listings
    dedup_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    audit["removed_records"]["duplicate_rows"] = dedup_count

    # Parse and clean target: selling_price
    price_col = None
    for cand in ["selling_price", "price", "target_price"]:
        if cand in cleaned.columns:
            price_col = cand
            break
            
    if price_col is None:
        raise ValueError("Target price column ('selling_price') not found in dataset.")

    cleaned["selling_price"] = cleaned[price_col].apply(extract_numeric_value)
    
    # Validate selling_price bounds (e.g., between ₹25,000 and ₹50,000,000)
    invalid_price_mask = (cleaned["selling_price"].isna()) | (cleaned["selling_price"] < 25000) | (cleaned["selling_price"] > 50000000)
    invalid_prices = int(invalid_price_mask.sum())
    cleaned = cleaned[~invalid_price_mask].reset_index(drop=True)
    audit["removed_records"]["invalid_prices"] = invalid_prices

    # Parse and clean year
    year_col = "year" if "year" in cleaned.columns else None
    if year_col:
        cleaned["year"] = pd.to_numeric(cleaned[year_col], errors="coerce")
        invalid_year_mask = (cleaned["year"].isna()) | (cleaned["year"] < 1985) | (cleaned["year"] > reference_year)
        invalid_years = int(invalid_year_mask.sum())
        cleaned = cleaned[~invalid_year_mask].reset_index(drop=True)
        audit["removed_records"]["invalid_years"] = invalid_years
    else:
        audit["removed_records"]["invalid_years"] = 0

    # Parse and clean km_driven
    km_col = None
    for cand in ["km_driven", "kms_driven", "mileage_km"]:
        if cand in cleaned.columns:
            km_col = cand
            break
            
    if km_col:
        cleaned["km_driven"] = cleaned[km_col].apply(extract_numeric_value)
        invalid_km_mask = (cleaned["km_driven"].isna()) | (cleaned["km_driven"] < 0) | (cleaned["km_driven"] > 1000000)
        invalid_km = int(invalid_km_mask.sum())
        cleaned = cleaned[~invalid_km_mask].reset_index(drop=True)
        audit["removed_records"]["invalid_km_driven"] = invalid_km
    else:
        audit["removed_records"]["invalid_km_driven"] = 0

    # Parse engine capacity (CC)
    if "engine" in cleaned.columns:
        cleaned["engine_cc"] = cleaned["engine"].apply(extract_numeric_value)
    elif "engine_cc" in cleaned.columns:
        cleaned["engine_cc"] = cleaned["engine_cc"].apply(extract_numeric_value)
    else:
        cleaned["engine_cc"] = np.nan

    # Parse mileage (fuel economy kmpl / km/kg)
    if "mileage" in cleaned.columns:
        cleaned["mileage_val"] = cleaned["mileage"].apply(extract_numeric_value)
    else:
        cleaned["mileage_val"] = np.nan

    # Parse max_power (bhp)
    if "max_power" in cleaned.columns:
        cleaned["max_power_bhp"] = cleaned["max_power"].apply(extract_numeric_value)
    else:
        cleaned["max_power_bhp"] = np.nan

    # Validate engine_cc bounds where present (typically 600cc to 6500cc)
    valid_engine_mask = (cleaned["engine_cc"].isna()) | ((cleaned["engine_cc"] >= 600) & (cleaned["engine_cc"] <= 6500))
    cleaned = cleaned[valid_engine_mask].reset_index(drop=True)

    # Impute missing numericals with median (grouped by brand if possible, else overall)
    num_cols_to_impute = ["engine_cc", "mileage_val", "max_power_bhp", "seats"]
    for col in num_cols_to_impute:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            med_val = cleaned[col].median()
            if pd.notna(med_val):
                cleaned[col] = cleaned[col].fillna(med_val)

    # Categorical normalization
    if "fuel" in cleaned.columns:
        cleaned["fuel"] = cleaned["fuel"].astype(str).str.strip().str.title()
        # Consolidate rare fuels if needed
        valid_fuels = ["Diesel", "Petrol", "Cng", "Lpg"]
        cleaned["fuel"] = cleaned["fuel"].apply(lambda f: f if f in valid_fuels else "Other")

    if "transmission" in cleaned.columns:
        cleaned["transmission"] = cleaned["transmission"].astype(str).str.strip().str.title()
        valid_trans = ["Manual", "Automatic"]
        cleaned["transmission"] = cleaned["transmission"].apply(lambda t: t if t in valid_trans else "Manual")

    if "seller_type" in cleaned.columns:
        cleaned["seller_type"] = cleaned["seller_type"].astype(str).str.strip().str.title()

    if "owner" in cleaned.columns:
        cleaned["owner"] = cleaned["owner"].astype(str).str.strip().str.title()

    audit["final_record_count"] = len(cleaned)
    audit["retention_rate_pct"] = round((len(cleaned) / initial_records) * 100, 2)
    audit["summary_statement"] = (
        f"Validation complete: Kept {len(cleaned)} of {initial_records} records "
        f"({audit['retention_rate_pct']}% retention). "
        f"Removed {dedup_count} duplicates and "
        f"{audit['removed_records']['invalid_prices']} invalid prices."
    )

    return cleaned, audit
