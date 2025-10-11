# tests/test_preprocess.py
import pandas as pd

from data_pipeline.preprocess import basic_impute, normalize_columns


def test_normalize_columns():
    df = pd.DataFrame({" A ": [1, 2]})
    out = normalize_columns(df)
    assert "A" in out.columns


def test_basic_impute():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "y"]})
    out = basic_impute(df)
    assert out["a"].isnull().sum() == 0
    assert out["b"].isnull().sum() == 0
