from pathlib import Path

import pandas as pd


def test_sample_readable():
    p = Path("data")
    assert p.exists()
