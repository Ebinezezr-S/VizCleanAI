# data_pipeline/preprocess.py
from typing import Tuple

import pandas as pd


def read_file(path: str) -> pd.DataFrame:
    """Read CSV/Excel with basic fallbacks."""
    if str(path).lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def basic_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            try:
                df[col] = df[col].fillna(df[col].median())
            except Exception:
                df[col] = df[col].fillna(0)
    return df


def clean_pipeline(path: str) -> Tuple[pd.DataFrame, dict]:
    df = read_file(path)
    df = normalize_columns(df)
    before = {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": df.isnull().sum().to_dict(),
    }
    df = basic_impute(df)
    after = {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": df.isnull().sum().to_dict(),
    }
    meta = {"before": before, "after": after}
    return df, meta
