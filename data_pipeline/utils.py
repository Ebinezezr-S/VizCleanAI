# data_pipeline/utils.py
import os
import uuid
from pathlib import Path


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_upload_path(base: str, filename: str) -> str:
    ensure_dir(base)
    uid = uuid.uuid4().hex[:8]
    return os.path.join(base, f"{uid}_{filename}")
