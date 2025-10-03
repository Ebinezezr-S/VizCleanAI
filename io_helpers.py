# io_helpers.py
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, IO

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def read_uploaded_file(fileobj: IO, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[Exception]]:
    """
    Read an uploaded file (stream or path-like). Returns (df, error).
    Supports CSV and Excel.
    """
    if fileobj is None:
        return None, ValueError("fileobj is None")

    try:
        name_lower = filename.lower()
        # Ensure stream is at start if it has seek
        try:
            fileobj.seek(0)
        except Exception:
            pass

        if name_lower.endswith((".xls", ".xlsx")):
            df = pd.read_excel(fileobj)
        else:
            # Try default, fallback to utf-8-sig on failure
            try:
                df = pd.read_csv(fileobj)
            except Exception:
                fileobj.seek(0)
                df = pd.read_csv(fileobj, encoding="utf-8-sig")
        return df, None
    except Exception as e:
        return None, e

def save_local_copy(df: pd.DataFrame, original_name: str) -> str:
    """
    Save DataFrame to DATA_DIR with timestamped safe name. Returns saved path.
    If original_name ends with .xls/.xlsx, saved as that extension; otherwise as .csv.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in (" ", ".", "_", "-") else "_" for c in original_name)
    base, ext = os.path.splitext(safe_name)
    if ext.lower() not in (".xls", ".xlsx", ".csv"):
        ext = ".csv"
    filename = f"{base}_{timestamp}{ext}"
    path = os.path.join(DATA_DIR, filename)

    # Save appropriately
    if ext.lower() in (".xls", ".xlsx"):
        # pandas will choose writer; saved as xlsx
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

    return path
