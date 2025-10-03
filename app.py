# main_app.py
"""
VizClean AI — Streamlit demo app (single-file)

Features:
 - Upload CSV (with latin1 fallback)
 - Basic cleaning: drop duplicates, fill numeric NA with column mean, fill non-numeric NA with ""
 - Show original + cleaned (head and full describe)
 - Outlier detection (numeric cols ±3 std)
 - Histogram visualization for selected numeric columns
 - Save cleaned file with timestamp and copy into ./deploy
 - Download cleaned .xlsx from UI
 - Uses session_state so reruns are safe
"""

from pathlib import Path
import shutil
from datetime import datetime
import io

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Page config
st.set_page_config(page_title="✨ VizClean AI ✨", layout="wide")
st.title("✨ VizClean AI ✨")
st.markdown("Transform raw data into clean insights in seconds")

# Paths
ROOT = Path(__file__).parent.resolve()
DEPLOY_DIR = ROOT / "deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

# Session state defaults
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "last_clean_path" not in st.session_state:
    st.session_state.last_clean_path = None

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # read CSV with fallback encoding
    try:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Original Data")
    st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
    st.dataframe(df.head(200))

    # store original in session
    st.session_state.df = df.copy()

    # Cleaning step (inside try/except)
    st.subheader("Cleaned Data")
    try:
        df_clean = df.drop_duplicates().copy()

        # Numeric fill
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            means = df_clean[numeric_cols].mean()
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(means)

        # Non-numeric fill
        non_numeric = df_clean.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            df_clean[non_numeric] = df_clean[non_numeric].fillna("")

        st.session_state.df_clean = df_clean
        st.dataframe(df_clean.head(200))
    except Exception as e:
        st.error(f"Cleaning failed: {e}")
        st.stop()

    # Data summary
    st.subheader("Data Summary")
    try:
        st.dataframe(df_clean.describe(include="all").T)
    except Exception as e:
        st.warning(f"Summary failed: {e}")

    # Auto insights: missing + outliers
    st.subheader("Auto Insights")
    try:
        missing = int(df_clean.isnull().sum().sum())
        st.write(f"Total missing values after cleaning: {missing}")
    except Exception:
        st.write("Total missing values after cleaning: (calculation failed)")

    with st.expander("Top rows with outliers (numeric columns ±3 std)"):
        try:
            numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
            outliers = pd.DataFrame()
            for col in numeric_cols:
                col_std = df_clean[col].std()
                col_mean = df_clean[col].mean()
                if pd.isna(col_std) or col_std == 0:
                    continue
                mask = (df_clean[col] - col_mean).abs() > 3 * col_std
                if mask.any():
                    outliers = pd.concat([outliers, df_clean[mask]])
            if outliers.empty:
                st.write("No outliers found (±3σ).")
            else:
                st.dataframe(outliers.drop_duplicates().head(200))
        except Exception as e:
            st.warning(f"Outlier detection failed: {e}")

    # Visualizations
    st.subheader("Visualizations")
    try:
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            cols_to_plot = st.multiselect("Select numeric columns to plot (histogram)", numeric_cols, default=numeric_cols[:2])
            for col in cols_to_plot:
                try:
                    fig, ax = plt.subplots()
                    ax.hist(df_clean[col].dropna(), bins=20)
                    ax.set_title(f"Histogram of {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot {col}: {e}")
        else:
            st.info("No numeric columns available for plotting.")
    except Exception as e:
        st.warning(f"Visualization section failed: {e}")

    # --- Save cleaned file and copy to deploy (guarded) ---
    st.subheader("Download Cleaned Data")
    if st.session_state.get("df_clean") is None:
        st.info("No cleaned data available to save. Upload and clean a CSV first.")
    else:
        try:
            timestamp = int(pd.Timestamp.utcnow().timestamp())
            clean_filename = f"cleaned_data_{timestamp}.xlsx"
            clean_path = ROOT / clean_filename

            # Save to root
            st.session_state.df_clean.to_excel(clean_path, index=False)

            # Copy to deploy
            DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clean_path, DEPLOY_DIR / clean_filename)

            st.session_state.last_clean_path = clean_path
            st.success(f"✅ Cleaned data saved as {clean_filename} (also copied to deploy/)")

            # Download button
            with open(clean_path, "rb") as fh:
                data_bytes = fh.read()
            st.download_button(
                label="Download cleaned .xlsx",
                data=data_bytes,
                file_name=clean_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Failed to save cleaned file: {e}")

else:
    st.info("Upload a CSV file to start. Example: ai4i2020.csv")

# Footer tips
st.markdown("---")
st.markdown(
    "Tips: Cleaned files saved to the project root as `cleaned_data_<ts>.xlsx` and copied to `deploy/` for serving. "
    "If you want to serve files programmatically, run a small FastAPI file server (e.g. `uvicorn file_server:api --host 127.0.0.1 --port 8502 --reload`)."
)
