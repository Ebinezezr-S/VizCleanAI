# data_pipeline/report.py
from typing import Dict

import pandas as pd


def generate_text_report(meta: Dict, summary_df: pd.DataFrame) -> str:
    lines = []
    lines.append("DATA QUALITY REPORT")
    lines.append("====================")
    lines.append("\nSUMMARY BEFORE:")
    lines.append(str(meta.get("before")))
    lines.append("\nSUMMARY AFTER:")
    lines.append(str(meta.get("after")))
    lines.append("\nCOLUMN SUMMARY:")
    lines.append(summary_df.to_string(index=False))
    return "\n".join(lines)
