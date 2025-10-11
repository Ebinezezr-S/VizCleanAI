# data_pipeline/eda.py
import pandas as pd


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        cols.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "n_missing": int(df[c].isnull().sum()),
                "n_unique": int(df[c].nunique(dropna=True)),
                "sample_values": df[c].dropna().unique()[:5].tolist(),
            }
        )
    return pd.DataFrame(cols)


def missing_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (df.isnull().sum() / len(df)).rename("missing_pct").to_frame()
