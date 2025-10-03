# main_app.py
"""
VizClean Streamlit app (full file)
- Upload CSV
- Optional drop_duplicates control
- Basic cleaning (customize clean_data)
- Save cleaned file to project root and copy into deploy/
- Show original/cleaned summaries and improved histogram visualization
- Provide download buttons for files in deploy/
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import shutil
import os
import altair as alt

# --- Config ---
st.set_page_config(page_title="? VizClean AI ?", layout="wide")
ROOT = Path(__file__).parent.resolve()
DEPLOY_DIR = ROOT / "deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers ---
def timestamp():
    return str(int(time.time()))

@st.cache_data
def read_csv_safe(uploaded_file):
    # robust CSV read (tries utf-8, then latin-1)
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")

def clean_data(df: pd.DataFrame, drop_duplicates_flag: bool = True) -> pd.DataFrame:
    """
    Minimal cleaning pipeline. Customize this as needed.
    Steps:
    - Optionally drop exact-duplicate rows
    - Strip whitespace from object (string) columns
    - Fill numeric NaNs with column median
    """
    df = df.copy()

    # 0. Optional: drop duplicates
    if drop_duplicates_flag:
        df = df.drop_duplicates().reset_index(drop=True)

    # 1. Trim whitespace in string/object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # 2. Fill numeric NaNs with column median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            median = df[c].median()
            df[c] = df[c].fillna(median)

    return df

def save_cleaned(df: pd.DataFrame) -> Path:
    ts = timestamp()
    fname = f"cleaned_data_{ts}.xlsx"
    out_path = ROOT / fname
    df.to_excel(out_path, index=False, engine="openpyxl")
    # copy to deploy
    shutil.copy2(out_path, DEPLOY_DIR / fname)
    return out_path

def get_deploy_files():
    if not DEPLOY_DIR.exists():
        return []
    files = sorted([p for p in DEPLOY_DIR.iterdir() if p.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
    return files

# --- UI ---
st.title("? VizClean AI ?")
st.write("Transform raw data into clean insights in seconds")

# Sidebar options
st.sidebar.header("Upload & Options")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)
drop_dup_checkbox = st.sidebar.checkbox("Drop exact duplicate rows", value=True)
fill_na_choice = st.sidebar.selectbox("If numeric missing: fill with", ["median", "0", "leave"], index=0)
show_preview = st.sidebar.checkbox("Show previews", value=True)

# If no file uploaded, show available cleaned files to download
if uploaded is None:
    st.info("Upload a CSV file to start. Example: ai4i2020.csv")
    st.markdown("**Available cleaned files (deploy/):**")
    files = get_deploy_files()
    if files:
        for f in files:
            with open(f, "rb") as fh:
                st.download_button(
                    label=f"Download {f.name}", 
                    data=fh, 
                    file_name=f.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.write("No cleaned files yet.")
    st.stop()

# Read uploaded CSV
try:
    df_orig = read_csv_safe(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.subheader("Original Data")
st.write(f"File: **{uploaded.name}**")
st.write(f"Rows: **{len(df_orig):,}** — Columns: **{df_orig.shape[1]}**")
if show_preview:
    st.dataframe(df_orig.head(5), width="stretch")

# Run cleaning
with st.spinner("Cleaning data..."):
    df_clean = clean_data(df_orig, drop_duplicates_flag=drop_dup_checkbox)

    # handle fill_na_choice for numeric columns
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    if fill_na_choice == "median":
        for c in num_cols:
            if df_clean[c].isna().any():
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    elif fill_na_choice == "0":
        for c in num_cols:
            if df_clean[c].isna().any():
                df_clean[c] = df_clean[c].fillna(0)
    # leave -> no-op

st.success("Cleaning complete ?")

# Save cleaned and copy to deploy
out_path = save_cleaned(df_clean)
st.info(f"? Cleaned data saved as **{out_path.name}** (also copied to deploy/)")

# Cleaned Data section
st.subheader("Cleaned Data")
st.write(f"Rows: **{len(df_clean):,}** — Columns: **{df_clean.shape[1]}**")
if show_preview:
    st.dataframe(df_clean.head(10), width="stretch")

# Data Summary
st.subheader("Data Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total missing values after cleaning", int(df_clean.isna().sum().sum()))
with col2:
    num_outliers = 0
    if len(num_cols) > 0:
        for c in num_cols:
            s = df_clean[c].dropna()
            if len(s) > 0:
                z = (s - s.mean()) / s.std(ddof=0)
                num_outliers += int((z.abs() > 3).sum())
    st.metric("Approx outlier count (±3s)", num_outliers)
with col3:
    st.metric("Columns", df_clean.shape[1])

# Summary statistics panel (toggleable)
if st.checkbox("Show numeric summary (describe)", value=False):
    if len(num_cols) > 0:
        st.dataframe(df_clean[num_cols].describe().T, width="stretch")
    else:
        st.info("No numeric columns to describe.")

# Improved Visualizations (histogram with bins)
st.subheader("Visualizations")
if len(num_cols) == 0:
    st.info("No numeric columns available for histogram plotting.")
else:
    choice = st.selectbox("Select numeric column to plot (histogram)", options=list(num_cols))
    if choice:
        st.write(f"Histogram of **{choice}** (binned)")
        arr = df_clean[choice].dropna().to_numpy()
        if arr.size == 0:
            st.info("No data to plot.")
        else:
            bins = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
            hist_counts, bin_edges = np.histogram(arr, bins=bins)
            bin_lefts = bin_edges[:-1]
            bin_rights = bin_edges[1:]
            mids = (bin_lefts + bin_rights) / 2
            hist_df = pd.DataFrame({
                "mid": mids,
                "count": hist_counts,
                "left": bin_lefts,
                "right": bin_rights
            })
            chart = (
                alt.Chart(hist_df)
                .mark_bar()
                .encode(
                    x=alt.X("mid:Q", title=choice, axis=alt.Axis(format="~s")),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=[alt.Tooltip("left:Q", title="bin left"), alt.Tooltip("right:Q", title="bin right"), alt.Tooltip("count:Q")]
                )
                .properties(height=330, width="container")
            )
            st.altair_chart(chart, use_container_width=True)

# Auto Insights
st.subheader("Auto Insights")
insights = []
if len(num_cols) > 0:
    variances = df_clean[num_cols].var(axis=0).sort_values(ascending=False)
    top_vars = variances.head(3).index.tolist()
    insights.append(f"Top variance numeric columns: {', '.join(top_vars)}")
no_missing = [c for c in df_clean.columns if df_clean[c].isna().sum() == 0]
if len(no_missing) > 0:
    insights.append(f"Columns with no missing values: {', '.join(no_missing[:6])}" + ("" if len(no_missing) <= 6 else ", ..."))
for i in insights:
    st.write("- " + i)

# Footer
st.markdown("---")
st.caption("Tips: Cleaned files saved to the project root as cleaned_data_<ts>.xlsx and copied to deploy/ for serving.")

