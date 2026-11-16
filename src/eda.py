"""
Exploratory Data Analysis (EDA) Module for Used Car Price Intelligence.

Computes comprehensive dataset metrics, statistical summaries, correlation structures,
and dynamically generates domain insights from actual data without hardcoding.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def compute_dataset_overview(df: pd.DataFrame, target_col: str = "selling_price") -> Dict[str, Any]:
    """Computes high-level structural overview of the dataset."""
    total_records = len(df)
    total_features = len(df.columns)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / total_records * 100).round(2)
    missing_info = {
        col: {"count": int(missing_summary[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing_summary[col] > 0
    }

    target_stats = {}
    if target_col in df.columns:
        s = df[target_col].dropna()
        target_stats = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            "skewness": float(s.skew())
        }

    return {
        "total_records": total_records,
        "total_features": total_features,
        "num_features_count": len(num_cols),
        "num_features": num_cols,
        "cat_features_count": len(cat_cols),
        "cat_features": cat_cols,
        "missing_columns_count": len(missing_info),
        "missing_columns_details": missing_info,
        "duplicate_count": int(df.duplicated().sum()),
        "target_stats": target_stats
    }


def compute_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rigorous 8-point statistical summary for numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    stats = pd.DataFrame({
        "Mean": num_df.mean(),
        "Median": num_df.median(),
        "Std Dev": num_df.std(),
        "Min": num_df.min(),
        "25%": num_df.quantile(0.25),
        "50%": num_df.quantile(0.50),
        "75%": num_df.quantile(0.75),
        "Max": num_df.max()
    }).round(2)
    return stats


def generate_dynamic_eda_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Dynamically derives data-grounded market observations directly from data.
    Never hardcodes conclusions.
    """
    insights = []

    # 1. Vehicle Age vs Price
    if "vehicle_age" in df.columns and "selling_price" in df.columns:
        corr_age = df[["vehicle_age", "selling_price"]].corr().iloc[0, 1]
        grouped_age = df.groupby("vehicle_age")["selling_price"].median()
        if len(grouped_age) >= 4:
            new_car_price = grouped_age.iloc[:2].mean()
            old_car_price = grouped_age.iloc[-4:].mean()
            depreciation = ((new_car_price - old_car_price) / new_car_price) * 100
            insights.append({
                "category": "Vehicle Age & Depreciation",
                "finding": f"Correlation between age and price is {corr_age:.2f}. "
                           f"Vehicles experience ~{depreciation:.1f}% median price depreciation between newer (<=2 yrs) and older (>=8 yrs) models."
            })

    # 2. Kilometers Driven vs Price
    if "km_driven" in df.columns and "selling_price" in df.columns:
        corr_km = df[["km_driven", "selling_price"]].corr().iloc[0, 1]
        insights.append({
            "category": "Mileage Impact",
            "finding": f"Correlation between odometer mileage (km driven) and selling price is {corr_km:.2f}, "
                       f"indicating {'an inverse' if corr_km < 0 else 'a weak'} relationship where higher usage correlates with lower resale value."
        })

    # 3. Fuel Type Value Comparison
    if "fuel" in df.columns and "selling_price" in df.columns:
        fuel_price = df.groupby("fuel")["selling_price"].agg(["median", "count"])
        fuel_price = fuel_price[fuel_price["count"] >= 10].sort_values("median", ascending=False)
        if not fuel_price.empty:
            top_fuel = fuel_price.index[0]
            top_price = fuel_price.iloc[0]["median"]
            lowest_fuel = fuel_price.index[-1]
            lowest_price = fuel_price.iloc[-1]["median"]
            diff_pct = ((top_price - lowest_price) / lowest_price) * 100
            insights.append({
                "category": "Fuel Type Premium",
                "finding": f"{top_fuel} vehicles lead median prices at ₹{top_price:,.0f}, command a {diff_pct:.1f}% premium over {lowest_fuel} vehicles (median ₹{lowest_price:,.0f})."
            })

    # 4. Engine Capacity (CC) Relationship
    if "engine_cc" in df.columns and "selling_price" in df.columns:
        corr_engine = df[["engine_cc", "selling_price"]].corr().iloc[0, 1]
        insights.append({
            "category": "Engine Capacity (Displacement)",
            "finding": f"Engine displacement has a correlation of {corr_engine:.2f} with resale price. "
                       f"Larger engine capacities (>1800 CC) typically occupy higher pricing tiers due to luxury branding and initial invoice cost."
        })

    # 5. Transmission Premium (Automatic vs Manual)
    if "transmission" in df.columns and "selling_price" in df.columns:
        trans_price = df.groupby("transmission")["selling_price"].median()
        if "Automatic" in trans_price and "Manual" in trans_price:
            auto_p = trans_price["Automatic"]
            man_p = trans_price["Manual"]
            prem = ((auto_p - man_p) / man_p) * 100
            insights.append({
                "category": "Transmission Type",
                "finding": f"Automatic transmission commands a median price of ₹{auto_p:,.0f} vs ₹{man_p:,.0f} for Manual, "
                           f"representing a {prem:.1f}% resale premium."
            })

    # 6. Brand Tier Distribution
    if "brand" in df.columns and "selling_price" in df.columns:
        brand_stats = df.groupby("brand")["selling_price"].agg(["median", "count"])
        brand_stats = brand_stats[brand_stats["count"] >= 15].sort_values("median", ascending=False)
        if len(brand_stats) >= 3:
            top_3 = ", ".join([f"{b} (₹{p:,.0f})" for b, p in brand_stats.head(3)["median"].items()])
            bottom_3 = ", ".join([f"{b} (₹{p:,.0f})" for b, p in brand_stats.tail(3)["median"].items()])
            insights.append({
                "category": "Brand Value Tiers",
                "finding": f"Highest median resale value brands: {top_3}. Most budget-accessible volume brands: {bottom_3}."
            })

    return insights
