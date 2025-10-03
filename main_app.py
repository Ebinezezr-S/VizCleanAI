import os
import io
import pandas as pd
import streamlit as st
from datetime import datetime

# Ensure data folder exists (ignored by git)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.title("VizClean — Upload & Preview")

# File uploader
uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded is None:
    st.info("Please upload a file to continue.")
else:
    # Show filename and size
    try:
        # uploaded may be a Streamlit UploadedFile object which supports .name and .size
        fname = uploaded.name
        fsize = getattr(uploaded, "size", None)
        if fsize:
            st.write(f"**File:** {fname} — {fsize//1024} KB")
        else:
            st.write(f"**File:** {fname}")
    except Exception:
        # defensive: sometimes UploadedFile differs by Streamlit version
        st.write(f"**File uploaded**")

    # Read file into a DataFrame with basic format detection
    df = None
    read_error = None
    try:
        name_lower = fname.lower()
        uploaded.seek(0)
        if name_lower.endswith((".xls", ".xlsx")):
            # For Excel files
            df = pd.read_excel(uploaded)
        else:
            # For CSV and generic text files
            # Try with default encoding; fallback to utf-8-sig if needed
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding="utf-8-sig")
    except Exception as e:
        read_error = e
        st.error(f"Failed to read uploaded file: {e}")

    # If read succeeded, present preview and save copy
    if df is not None:
        st.success("File read successfully.")
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df.head())

        # Save a timestamped copy in data/ for local usage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in (" ", ".", "_", "-") else "_" for c in fname)
        save_name = f"{os.path.splitext(safe_name)[0]}_{timestamp}{os.path.splitext(safe_name)[1]}"
        save_path = os.path.join(DATA_DIR, save_name)

        # Save as original format when possible; for csv fallback to csv
        try:
            if name_lower.endswith((".xls", ".xlsx")):
                df.to_excel(save_path, index=False)
            else:
                # ensure .csv extension
                if not save_path.lower().endswith(".csv"):
                    save_path = os.path.splitext(save_path)[0] + ".csv"
                df.to_csv(save_path, index=False)
            st.write(f"Saved a local copy to `{save_path}` (not tracked by git).")
        except Exception as e:
            st.warning(f"Could not save local copy: {e}")

        # Store DataFrame in session state for downstream processing
        st.session_state["uploaded_df"] = df

        # Optional: a separate button to run heavier cleaning/processing
        if st.button("Run cleaning pipeline"):
            st.info("Running cleaning steps...")
            try:
                # Example cleaning steps (customize to your pipeline)
                # 1) Drop fully empty rows
                cleaned = df.dropna(how="all").copy()

                # 2) Trim whitespace from string columns
                for c in cleaned.select_dtypes(include=["object"]).columns:
                    cleaned[c] = cleaned[c].astype(str).str.strip()

                # 3) Show result summary
                st.subheader("Cleaning result (sample)")
                st.dataframe(cleaned.head())

                # Save cleaned to data/ with suffix
                cleaned_name = os.path.splitext(save_path)[0] + "_cleaned.csv"
                cleaned.to_csv(cleaned_name, index=False)
                st.success(f"Cleaning finished. Cleaned file saved to `{cleaned_name}`.")
                st.session_state["cleaned_df"] = cleaned
            except Exception as e:
                st.error(f"Cleaning pipeline failed: {e}")

    else:
        st.error("No dataframe to show due to read error. Please check the file format.")
