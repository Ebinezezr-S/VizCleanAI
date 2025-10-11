# data_pipeline/transform.py
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_numeric(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
