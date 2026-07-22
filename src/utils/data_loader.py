import pandas as pd
from pathlib import Path

# Base project directory
BASE_DIR = Path("D:/machinelearning/houseprice_prediction_project")

def load_processed(path=BASE_DIR / "data/processed/housing_cleaned.csv"):
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Could not find file at {path}")
    return pd.read_csv(path)

def load_raw(path="data/raw/housing.csv"):
    return pd.read_csv(path)


