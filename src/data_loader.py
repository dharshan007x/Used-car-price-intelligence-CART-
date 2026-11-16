"""
Data Loader Module for Used Car Price Intelligence System.

Handles loading datasets from local storage, remote mirrors, or user uploads.
Ensures graceful error handling, fallback URLs, and format validation.
"""

from pathlib import Path
import logging
import urllib.request
from typing import Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATASET_URLS = [
    "https://raw.githubusercontent.com/HarshKapadia2/car-details/main/data/car_details_v3.csv",
    "https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv",
    "https://raw.githubusercontent.com/neelgandhi108/Car-Price-Prediction/master/car%20data.csv"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "car_details_v3.csv"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_car_data.csv"


def download_default_dataset(target_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Downloads the CarDekho dataset from public repository mirrors if not present locally.
    """
    target = Path(target_path) if target_path else DEFAULT_RAW_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 1000:
        logger.info(f"Dataset already exists at {target}")
        return target

    last_error = None
    for url in DEFAULT_DATASET_URLS:
        try:
            logger.info(f"Attempting to download dataset from {url}...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(target, 'wb') as out_file:
                out_file.write(response.read())
            if target.stat().st_size > 1000:
                logger.info(f"Successfully downloaded dataset to {target} ({target.stat().st_size} bytes)")
                return target
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            last_error = e

    raise RuntimeError(
        f"Unable to download used car dataset from any remote mirror. "
        f"Please manually place 'car_details_v3.csv' in '{target.parent}'. Last error: {last_error}"
    )


def load_raw_dataset(
    file_path: Optional[Union[str, Path]] = None,
    auto_download: bool = True
) -> pd.DataFrame:
    """
    Loads raw CSV dataset into a pandas DataFrame.
    Supports auto-download if the file is missing.
    """
    path = Path(file_path) if file_path else DEFAULT_RAW_PATH

    if not path.exists():
        if auto_download:
            path = download_default_dataset(path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Set auto_download=True or specify valid file path."
            )

    # Try common encodings
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"Loaded {len(df)} records from {path} with {enc} encoding.")
            break
        except (UnicodeDecodeError, Exception) as e:
            continue

    if df is None:
        raise ValueError(f"Could not parse CSV at {path} with any supported encoding.")

    return df


def load_processed_dataset(
    file_path: Optional[Union[str, Path]] = None
) -> Optional[pd.DataFrame]:
    """
    Loads processed/cleaned dataset if available.
    """
    path = Path(file_path) if file_path else DEFAULT_PROCESSED_PATH
    if path.exists():
        return pd.read_csv(path)
    return None
