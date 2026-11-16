"""
Feature Engineering Module for Used Car Price Intelligence.

Extracts vehicle brands and models using a configurable alias-aware parser,
derives vehicle age, annual usage intensity, ownership tier, seller type,
trim tier, body segment, torque parsing, and market-style flags — all
without train-test leakage.
"""

from typing import Tuple, List, Optional, Dict
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_ALIAS_PATH = _CONFIG_DIR / "model_aliases.json"

_ALIAS_CONFIG: Optional[Dict] = None


def _load_alias_config() -> Dict:
    """Loads and caches the model alias configuration."""
    global _ALIAS_CONFIG
    if _ALIAS_CONFIG is not None:
        return _ALIAS_CONFIG
    if _ALIAS_PATH.exists():
        with open(_ALIAS_PATH, "r", encoding="utf-8") as f:
            _ALIAS_CONFIG = json.load(f)
    else:
        _ALIAS_CONFIG = {"aliases": {}, "brand_corrections": {},
                         "multi_word_brands": {}, "multi_word_models": {}}
    return _ALIAS_CONFIG


# ──────────────────────────────────────────────
# Brand & Model Extraction (Improved)
# ──────────────────────────────────────────────

def extract_brand_and_model(car_name: str) -> Tuple[str, str]:
    """
    Extracts brand and model from vehicle name string using the alias config.

    Handles multi-word brands (Land Rover, Mercedes-Benz, …),
    multi-word models (Swift Dzire, Grand i10, 3 Series, …),
    alias normalization (WagonR → Wagon R, New → C-Class, …),
    and correct casing (BMW not Bmw).

    Examples:
        'Maruti Swift Dzire VDI' -> ('Maruti', 'Swift Dzire')
        'BMW 3 Series 320d'      -> ('BMW', '3 Series')
        'Hyundai Grand i10 Magna' -> ('Hyundai', 'Grand i10')
        'Maruti Wagon R LXI'     -> ('Maruti', 'Wagon R')
        'Mercedes-Benz New C-Class 220' -> ('Mercedes-Benz', 'C-Class')
    """
    if pd.isna(car_name) or not str(car_name).strip():
        return "Unknown", "Unknown"

    cfg = _load_alias_config()
    brand_corrections = cfg.get("brand_corrections", {})
    multi_word_brands_raw = cfg.get("multi_word_brands", {})
    # Filter out metadata keys (prefixed with _)
    multi_word_brands = {k: v for k, v in multi_word_brands_raw.items()
                         if not k.startswith("_") and isinstance(v, int)}
    multi_word_models_raw = cfg.get("multi_word_models", {})
    multi_word_models = {k: v for k, v in multi_word_models_raw.items()
                         if not k.startswith("_") and isinstance(v, list)}
    aliases = cfg.get("aliases", {})

    raw = str(car_name).strip()
    parts = raw.split()
    if not parts:
        return "Unknown", "Unknown"

    # ── Step 1: Identify brand ────────────────────────────────
    brand = None
    brand_tokens = 1  # how many tokens the brand consumes

    # Check multi-word brands first (longest match)
    for mw_brand, token_count in sorted(multi_word_brands.items(),
                                         key=lambda x: -x[1]):
        if token_count == 1:
            continue  # handled below
        if len(parts) >= token_count:
            candidate = " ".join(parts[:token_count])
            if candidate.lower().replace("-", "").replace(" ", "") == mw_brand.lower().replace("-", "").replace(" ", ""):
                brand = mw_brand
                brand_tokens = token_count
                break

    # Single-token brand
    if brand is None:
        token0_lower = parts[0].lower()
        # Check if it's a hyphenated brand like Mercedes-Benz
        brand = brand_corrections.get(token0_lower, parts[0])
        brand_tokens = 1

    # ── Step 2: Remaining tokens after brand ──────────────────
    remaining = parts[brand_tokens:]
    if not remaining:
        return brand, "Standard"

    # ── Step 3: Try multi-word model match ────────────────────
    brand_mw_models = multi_word_models.get(brand, [])
    # Sort by number of tokens (longest first) for greedy match
    brand_mw_models_sorted = sorted(brand_mw_models, key=lambda m: -len(m.split()))

    model = None
    for mw_model in brand_mw_models_sorted:
        mw_parts = mw_model.split()
        n = len(mw_parts)
        if len(remaining) >= n:
            candidate = " ".join(remaining[:n])
            if candidate.lower() == mw_model.lower():
                model = mw_model  # use canonical casing from config
                break

    # ── Step 4: Fallback — single-token model ─────────────────
    if model is None:
        model_token = remaining[0]
        model_key = model_token.lower()
        # Check alias table
        if model_key in aliases:
            model = aliases[model_key]
        else:
            model = model_token

    # ── Step 5: Final alias pass (catches compound like "New" for MB)
    model_lower = model.lower()
    if model_lower in aliases:
        model = aliases[model_lower]

    return brand, model


# ──────────────────────────────────────────────
# Brand-Model Map Builder (for UI dropdowns)
# ──────────────────────────────────────────────

def build_brand_model_map(
    df: pd.DataFrame,
    brand_col: str = "brand",
    model_col: str = "model_name",
) -> Dict[str, List[Dict[str, any]]]:
    """
    Builds a brand → list-of-models mapping from the FULL cleaned dataset.

    Each model entry has:
      - name: canonical model name
      - count: number of listings

    Models are sorted by count (descending), then alphabetically.
    Every (brand, model) pair present in the dataset is included —
    no min_frequency filter is applied here.

    This map is for UI dropdowns only and does not affect training.
    """
    if brand_col not in df.columns or model_col not in df.columns:
        return {}

    counts = df.groupby([brand_col, model_col]).size().reset_index(name="count")
    brand_map: Dict[str, List[Dict]] = {}

    for _, row in counts.iterrows():
        brand = str(row[brand_col])
        model = str(row[model_col])
        cnt = int(row["count"])
        if brand not in brand_map:
            brand_map[brand] = []
        brand_map[brand].append({"name": model, "count": cnt})

    # Sort each brand's models: by count desc, then alpha
    for brand in brand_map:
        brand_map[brand] = sorted(
            brand_map[brand],
            key=lambda m: (-m["count"], m["name"]),
        )

    return brand_map


# ──────────────────────────────────────────────
# Owner Tier
# ──────────────────────────────────────────────

OWNER_TIER_MAP: Dict[str, int] = {
    "first owner": 1,
    "second owner": 2,
    "third owner": 3,
    "fourth & above owner": 4,
    "fourth and above owner": 4,
    "test drive car": 5,
}


def encode_owner_tier(owner_str: str) -> int:
    """
    Converts owner string to ordinal integer.
    First=1, Second=2, Third=3, Fourth+=4, TestDrive=5, Unknown=2 (default).
    """
    if pd.isna(owner_str):
        return 2
    key = str(owner_str).strip().lower()
    return OWNER_TIER_MAP.get(key, 2)


# ──────────────────────────────────────────────
# Trim Tier Extraction
# ──────────────────────────────────────────────

TOP_TRIM_KEYWORDS = {
    "zxi+", "zdi+", "asta", "top", "prestige", "titanium+", "signature",
    "luxury", "z+", "alpha", "prime", "excellence", "legend", "platinum",
    "aura+", "touring", "gt", "s+", "anniversary", "limited",
    "special edition", "amg", "expert", "flagship",
}
BASE_TRIM_KEYWORDS = {
    "lxi", "ldi", "l", "e", "std", "d", "gl", "base", "entry",
    "s", "g", "ls", "lxe", "s1",
}


def extract_trim_tier(car_name: str) -> str:
    """
    Derives trim tier (Base/Mid/Top) from variant keywords in car name.
    Default is Mid for unrecognised variants.
    """
    if pd.isna(car_name):
        return "Mid"
    tokens = set(t.lower() for t in str(car_name).strip().split())
    if tokens & TOP_TRIM_KEYWORDS:
        return "Top"
    if tokens & BASE_TRIM_KEYWORDS:
        return "Base"
    return "Mid"


# ──────────────────────────────────────────────
# Body Segment Lookup
# ──────────────────────────────────────────────

BODY_SEGMENT_MAP: Dict[str, str] = {
    # Hatchbacks
    "swift": "Hatchback", "alto": "Hatchback", "alto 800": "Hatchback",
    "alto k10": "Hatchback", "wagon r": "Hatchback",
    "santro": "Hatchback", "santro xing": "Hatchback",
    "i10": "Hatchback", "i20": "Hatchback", "grand i10": "Hatchback",
    "elite i20": "Hatchback", "i20 active": "Hatchback",
    "polo": "Hatchback", "jazz": "Hatchback",
    "celerio": "Hatchback", "kwid": "Hatchback", "tiago": "Hatchback",
    "ignis": "Hatchback", "baleno": "Hatchback", "glanza": "Hatchback",
    "beat": "Hatchback", "brio": "Hatchback",
    "micra": "Hatchback", "pulse": "Hatchback", "eon": "Hatchback",
    "ritz": "Hatchback", "etios liva": "Hatchback",
    "a-star": "Hatchback", "spark": "Hatchback",
    "zen": "Hatchback", "redigo": "Hatchback", "go": "Hatchback",
    "go plus": "Hatchback", "matiz": "Hatchback", "palio": "Hatchback",
    "s-presso": "Hatchback", "figo": "Hatchback",
    "punto": "Hatchback", "grande punto": "Hatchback",
    "indica": "Hatchback", "triber": "Hatchback",
    "800": "Hatchback", "gti": "Hatchback", "crosspolo": "Hatchback",
    "avventura": "Hatchback", "astra": "Hatchback", "309": "Hatchback",
    # Sedans
    "swift dzire": "Sedan", "dzire": "Sedan", "city": "Sedan",
    "ciaz": "Sedan", "verna": "Sedan", "amaze": "Sedan",
    "aspire": "Sedan", "xcent": "Sedan", "etios": "Sedan",
    "rapid": "Sedan", "vento": "Sedan",
    "accent": "Sedan", "sonata": "Sedan", "camry": "Sedan",
    "corolla": "Sedan", "civic": "Sedan", "elantra": "Sedan",
    "jetta": "Sedan", "superb": "Sedan", "octavia": "Sedan",
    "accord": "Sedan", "linea": "Sedan", "manza": "Sedan",
    "sunny": "Sedan", "verito": "Sedan", "tigor": "Sedan",
    "zest": "Sedan", "ameo": "Sedan", "scala": "Sedan",
    "laura": "Sedan", "fiesta": "Sedan", "sx4": "Sedan",
    "classic": "Sedan", "indigo": "Sedan", "passat": "Sedan",
    "yaris": "Sedan", "sail": "Sedan", "aveo": "Sedan",
    "fluence": "Sedan", "lancer": "Sedan", "teana": "Sedan",
    "3 series": "Sedan", "5 series": "Sedan", "7 series": "Sedan",
    "c-class": "Sedan", "e-class": "Sedan", "s-class": "Sedan",
    "cla": "Sedan", "a3": "Sedan", "a4": "Sedan", "a6": "Sedan",
    "xf": "Sedan", "xe": "Sedan", "es": "Sedan",
    "s60": "Sedan", "s90": "Sedan", "v40": "Sedan",
    "b-class": "Sedan", "gla": "Sedan",
    # SUVs / Crossovers
    "creta": "SUV", "seltos": "SUV", "nexon": "SUV", "ecosport": "SUV",
    "vitara brezza": "SUV", "duster": "SUV", "kicks": "SUV",
    "terrano": "SUV", "harrier": "SUV", "safari": "SUV", "compass": "SUV",
    "xc40": "SUV", "xc60": "SUV", "xc90": "SUV", "fortuner": "SUV",
    "endeavour": "SUV", "tucson": "SUV", "cr-v": "SUV",
    "captiva": "SUV", "pajero": "SUV",
    "brezza": "SUV", "venue": "SUV", "sonet": "SUV",
    "thar": "SUV", "bolero": "SUV", "scorpio": "SUV",
    "xuv300": "SUV", "xuv500": "SUV", "xuv700": "SUV",
    "freestyle": "SUV", "punch": "SUV", "hector": "SUV",
    "kuv 100": "SUV", "nuvosport": "SUV", "quanto": "SUV",
    "hexa": "SUV", "trailblazer": "SUV",
    "wrangler": "SUV", "koleos": "SUV", "captur": "SUV",
    "x1": "SUV", "x3": "SUV", "x4": "SUV", "x5": "SUV", "x6": "SUV", "x7": "SUV",
    "q3": "SUV", "q5": "SUV", "q7": "SUV",
    "gl-class": "SUV", "glc": "SUV", "m-class": "SUV",
    "land cruiser": "SUV", "range rover": "SUV",
    "discovery": "SUV", "freelander": "SUV",
    "ssangyong rexton": "SUV", "d-max": "SUV", "mu-x": "SUV",
    "gurkha": "SUV", "yeti": "SUV", "kodiaq": "SUV",
    "6 series": "SUV",
    # MUVs / MPVs
    "innova": "MUV", "ertiga": "MUV", "marazzo": "MUV", "lodgy": "MUV",
    "enjoy": "MUV", "tavera": "MUV", "sumo": "MUV",
    "xl6": "MUV", "carnival": "MUV", "qualis": "MUV",
    "xylo": "MUV", "multivan": "MUV",
    # Vans
    "omni": "Van", "eeco": "Van",
    # Pickup
    "xenon": "Pickup",
    # Others
    "wr-v": "SUV", "br-v": "SUV",
}


def derive_body_segment(model_str: str) -> str:
    """
    Derives body segment from model name using documented lookup.
    Returns 'Sedan' as default for unmatched models.
    """
    if pd.isna(model_str):
        return "Sedan"
    key = str(model_str).strip().lower()
    return BODY_SEGMENT_MAP.get(key, "Sedan")


# ──────────────────────────────────────────────
# Torque Parsing
# ──────────────────────────────────────────────

def parse_torque_nm(torque_str: str) -> float:
    """
    Extracts torque in Nm from strings like '115Nm@ 4500rpm', '11.5kgm at 2500rpm'.
    Returns np.nan if not parseable.
    """
    if pd.isna(torque_str):
        return np.nan
    s = str(torque_str).strip()
    # Try Nm value
    nm_match = re.search(r"(\d+\.?\d*)\s*[Nn][Mm]", s)
    if nm_match:
        return float(nm_match.group(1))
    # Try kgm conversion (1 kgm ≈ 9.806 Nm)
    kgm_match = re.search(r"(\d+\.?\d*)\s*kgm", s, re.IGNORECASE)
    if kgm_match:
        return round(float(kgm_match.group(1)) * 9.806, 1)
    # Fallback: extract first number (sanity check 50-900)
    fallback = re.search(r"(\d+\.?\d*)", s)
    if fallback:
        val = float(fallback.group(1))
        if 50 <= val <= 900:
            return val
    return np.nan


# ──────────────────────────────────────────────
# Master Feature Engineering Function
# ──────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    reference_year: Optional[int] = None,
    min_brand_count: int = 5,
    typical_km_per_year: Optional[float] = None,
) -> pd.DataFrame:
    """
    Applies domain-specific feature engineering:
    - Extracts Brand and Model Name (alias-aware parser)
    - Derives Vehicle Age from Year
    - Derives Annual Usage Intensity (km_per_year)
    - km_ratio: km_driven / (vehicle_age * typical_km_per_year), age-normalised usage
    - Owner Tier (ordinal integer)
    - Trim Tier (Base/Mid/Top)
    - Body Segment (Hatchback/Sedan/SUV/MUV/Van)
    - Torque (Nm)
    - High-mileage flag
    - Engine capacity bins for EDA
    - Price category for stratification

    Args:
        typical_km_per_year: Median km/year from training partition only.
            Pass None to skip km_ratio (it will be set to 1.0).
            Must NOT be computed from test data to avoid leakage.
    """
    df_out = df.copy()

    # Determine reference year consistently
    if reference_year is None:
        if "year" in df_out.columns and not df_out["year"].dropna().empty:
            reference_year = int(df_out["year"].max()) + 1
        else:
            reference_year = 2024

    # ── Brand and Model ──
    name_col = None
    for cand in ["name", "car_name", "vehicle_name"]:
        if cand in df_out.columns:
            name_col = cand
            break

    if name_col and "brand" not in df_out.columns:
        brand_model = df_out[name_col].apply(extract_brand_and_model)
        df_out["brand"] = [bm[0] for bm in brand_model]
        df_out["model_name"] = [bm[1] for bm in brand_model]
    elif "brand" not in df_out.columns:
        df_out["brand"] = "Unknown"
        df_out["model_name"] = "Unknown"

    # Back-fill model_name if brand was already present but model_name absent
    if "model_name" not in df_out.columns and name_col:
        brand_model = df_out[name_col].apply(extract_brand_and_model)
        df_out["model_name"] = [bm[1] for bm in brand_model]

    # ── Vehicle Age ──
    if "year" in df_out.columns:
        df_out["vehicle_age"] = reference_year - df_out["year"]
        df_out["vehicle_age"] = df_out["vehicle_age"].clip(lower=0)

    # ── Annual Usage Intensity ──
    if "km_driven" in df_out.columns and "vehicle_age" in df_out.columns:
        effective_age = df_out["vehicle_age"].replace(0, 0.5)
        df_out["km_per_year"] = (df_out["km_driven"] / effective_age).round(1)

    # ── Owner Tier (ordinal) ──
    if "owner" in df_out.columns:
        df_out["owner_tier"] = df_out["owner"].apply(encode_owner_tier)
    else:
        df_out["owner_tier"] = 2

    # ── Trim Tier ──
    if name_col:
        df_out["trim_tier"] = df_out[name_col].apply(extract_trim_tier)
    else:
        df_out["trim_tier"] = "Mid"

    # ── Body Segment ──
    if "model_name" in df_out.columns:
        df_out["body_segment"] = df_out["model_name"].apply(derive_body_segment)
    else:
        df_out["body_segment"] = "Sedan"

    # ── Torque (Nm) ──
    if "torque" in df_out.columns:
        df_out["torque_nm"] = df_out["torque"].apply(parse_torque_nm)
    else:
        df_out["torque_nm"] = np.nan

    # ── High-mileage flag ──
    if "km_per_year" in df_out.columns:
        df_out["high_mileage_flag"] = (df_out["km_per_year"] > 20000).astype(int)
    else:
        df_out["high_mileage_flag"] = 0

    # ── km_ratio: age-normalised usage intensity ──
    # km_ratio = km_driven / expected_km   where expected_km = vehicle_age * typical_km_per_year
    # > 1.0 means more driven than typical; < 1.0 means less.
    # typical_km_per_year MUST be computed from training data only (no leakage).
    # At prediction time, the stored value from model_metadata.json is used.
    if "km_driven" in df_out.columns and "vehicle_age" in df_out.columns:
        _tkm = typical_km_per_year if (typical_km_per_year is not None and typical_km_per_year > 0) else 10000.0
        effective_age_for_ratio = df_out["vehicle_age"].clip(lower=0.5)
        expected_km = effective_age_for_ratio * _tkm
        df_out["km_ratio"] = (df_out["km_driven"] / expected_km).clip(lower=0.05, upper=10.0).round(4)
    else:
        df_out["km_ratio"] = 1.0

    # ── Engine Capacity Bin (for EDA) ──
    if "engine_cc" in df_out.columns:
        bins = [0, 1000, 1500, 2000, 10000]
        labels = ["< 1000 CC", "1000-1499 CC", "1500-1999 CC", ">= 2000 CC"]
        df_out["engine_segment"] = pd.cut(
            df_out["engine_cc"], bins=bins, labels=labels, right=False
        )

    # ── Price Segment (for stratification) ──
    if "selling_price" in df_out.columns:
        price_bins = [0, 300000, 700000, 1500000, 100000000]
        price_labels = [
            "Budget (< ₹3L)", "Mid-Range (₹3L-₹7L)",
            "Premium (₹7L-₹15L)", "Luxury (> ₹15L)",
        ]
        df_out["price_segment"] = pd.cut(
            df_out["selling_price"], bins=price_bins, labels=price_labels, right=False
        )

    return df_out
