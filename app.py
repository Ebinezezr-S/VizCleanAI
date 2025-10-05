import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

# --------------------------------------------------------
# ⚙️ App Config
# --------------------------------------------------------
st.set_page_config(page_title="✨ VizClean AI ✨", layout="wide")

st.title("✨ VizClean AI ✨")
st.subheader("Transform raw data into clean insights in seconds 🚀")

# --------------------------------------------------------
# 📤 File Upload
# --------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        st.warning("⚠️ Could not decode file using UTF-8 — retrying with ISO-8859-1")
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    st.success(f"✅ File uploaded successfully: **{uploaded_file.name}** ({len(df)} rows, {len(df.columns)} columns)")

    # Tabs
    tabs = st.tabs(["Original Data", "Cleaned Data", "Data Summary", "Auto Insights", "Visualizations"])

    # --------------------------------------------------------
    # 🧹 Cleaning Process
    # --------------------------------------------------------
    df_clean = df.copy()

    # Remove completely empty columns
    df_clean.dropna(axis=1, how="all", inplace=True)

    # Fill missing numeric values with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Fill missing categorical values with mode
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    total_missing = df_clean.isnull().sum().sum()

    # --------------------------------------------------------
    # 🧾 Tab 1: Original Data
    # --------------------------------------------------------
    with tabs[0]:
        st.write("### 🧾 Original Data Preview")
        st.dataframe(df.head(20))

    # --------------------------------------------------------
    # 🧽 Tab 2: Cleaned Data
    # --------------------------------------------------------
    with tabs[1]:
        st.write(f"### 🧽 Cleaned Data — Total missing values after cleaning: **{total_missing}**")
        st.dataframe(df_clean.head(20))

        # 🔽 Excel / CSV download (memory only)
        def make_excel_download(df: pd.DataFrame):
            try:
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)
                data = buffer.getvalue()
                filename = "cleaned_data.xlsx"
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                return data, filename, mime
            except Exception as e:
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                csv_buf.seek(0)
                data = csv_buf.getvalue().encode("utf-8")
                filename = "cleaned_data.csv"
                mime = "text/csv"
                st.warning(f"Excel export failed — providing CSV instead ({e}).")
                return data, filename, mime

        data, filename, mime = make_excel_download(df_clean)
        st.download_button("📥 Download Cleaned Data", data=data, file_name=filename, mime=mime)

    # --------------------------------------------------------
    # 📊 Tab 3: Data Summary
    # --------------------------------------------------------
    with tabs[2]:
        st.write("### 📊 Data Summary")
        st.write(df_clean.describe(include="all").transpose())

    # --------------------------------------------------------
    # 🔍 Tab 4: Auto Insights
    # --------------------------------------------------------
    with tabs[3]:
        st.write("### 🤖 Auto Insights")
        st.write("Top 5 rows with potential outliers (numeric columns ±3 std):")

        if not numeric_cols.empty:
            z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std())
            outlier_mask = (z_scores > 3).any(axis=1)
            outliers = df_clean[outlier_mask].head(5)
            st.dataframe(outliers)
        else:
            st.info("No numeric columns available for outlier detection.")

    # --------------------------------------------------------
    # 📈 Tab 5: Visualizations
    # --------------------------------------------------------
    with tabs[4]:
        st.write("### 📈 Visualizations")
        cols = st.multiselect("Select columns to plot", df_clean.columns.tolist())

        if len(cols) == 1:
            st.bar_chart(df_clean[cols])
        elif len(cols) == 2:
            fig = px.scatter(df_clean, x=cols[0], y=cols[1], title=f"{cols[0]} vs {cols[1]}")
            st.plotly_chart(fig, use_container_width=True)
        elif len(cols) > 2:
            st.warning("Please select only 1 or 2 columns for visualization.")
        else:
            st.info("Select at least one column to visualize.")
else:
    st.info("📤 Upload a CSV file to get started.")

