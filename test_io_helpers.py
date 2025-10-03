import io
import pandas as pd
from io_helpers import read_uploaded_file, save_local_copy, DATA_DIR
from pathlib import Path

def test_read_csv():
    s = "a,b\n1,2\n3,4\n"
    f = io.StringIO(s)
    df, err = read_uploaded_file(f, "sample.csv")
    assert err is None
    assert df.shape == (2,2)

def test_save_local_copy(tmp_path):
    # create a small df
    df = pd.DataFrame({"x":[1,2], "y":[3,4]})
    # monkeypatch DATA_DIR by writing to tmp_path
    p = tmp_path / "data"
    p.mkdir()
    # temporarily set DATA_DIR path by calling function that respects env var (we didn't implement var)
    # Simpler: just call and check file exists in data/
    saved = save_local_copy(df, "test.csv")
    assert Path(saved).exists()
    # cleanup saved
    Path(saved).unlink()
