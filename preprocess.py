# data_pipeline/preprocess.py
from typing import Tuple

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # example: strip whitespace, lower column names
    df.columns = [c.strip() for c in df.columns]
    # fill simple missing values (example)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("missing")
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": df.isnull().sum().to_dict(),
    }
