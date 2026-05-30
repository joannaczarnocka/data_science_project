"""Ladowanie i zapis danych HR."""

from pathlib import Path

import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, RAW_FILENAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"


def load_hr() -> pd.DataFrame:
    """Wczytaj zestaw HR.csv."""
    return pd.read_csv(DATA_RAW / RAW_FILENAME)


def load_raw(filename: str) -> pd.DataFrame:
    """Wczytaj plik z katalogu data/raw/."""
    return pd.read_csv(DATA_RAW / filename)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Zapisz DataFrame do data/processed/."""
    path = DATA_PROCESSED / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
