# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="VizClean AI — Local", layout="wide")

st.title("✨ VizClean AI — Local (Updated)")

# ---- Upload / load ----
uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="Limit file size by Streamlit config")
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    # optionally, if a default local CSV exists (for dev), load it:
    if st.checkbox("Load example: diabetes.csv from project folder"):
        try:
            df = pd.read_csv("data/diabetes.csv")
            st.success("Loaded data/diabetes.csv")
        except Exception as e:
            st.error(f"Couldn't load example: {e}")

if df is None:
    st.info("Upload a CSV to begin. Or enable example load.")
    st.stop()

# show basic info
st.markdown(f"**File:** `{getattr(uploaded, 'name', 'local file')}` — Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.subheader("Original Data — First 10 rows")
st.dataframe(df.head(10), height=240)

# ---- Data summary ----
st.subheader("Dataset Summary")
with st.expander("Basic summary"):
    buffer = BytesIO()
    st.write("Columns:", list(df.columns))
    st.write("Dtypes:")
    st.write(df.dtypes)
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

# numeric summary
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 0:
    st.write("Numeric summary (first 10 columns shown):")
    st.dataframe(df[num_cols].describe().transpose().round(4))

# ---- Imputation / cleaning ----
st.subheader("🧽 Fill Missing Values (Imputation)")

col_to_impute = st.multiselect("Columns to impute (leave empty = all numeric columns)", options=df.columns.tolist())

numeric_strategy = st.selectbox("Numeric strategy", ["mean", "median", "constant"])
numeric_constant = None
if numeric_strategy == "constant":
    numeric_constant = st.number_input("Numeric constant value", value=0.0)

text_strategy = st.selectbox("Text strategy", ["mode", "constant"])
text_constant = None
if text_strategy == "constant":
    text_constant = st.text_input("Text constant value", value="")

apply_impute = st.button("Apply imputation (preview only)")

# create cleaned copy
df_clean = df.copy()

if apply_impute:
    # choose columns
    if not col_to_impute:
        # default: all columns with any missing values
        col_to_impute = df_clean.columns[df_clean.isnull().any()].tolist()

    for col in col_to_impute:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            if numeric_strategy == "mean":
                fillval = df_clean[col].mean()
            elif numeric_strategy == "median":
                fillval = df_clean[col].median()
            else:
                fillval = numeric_constant
            # SAFE fill (avoid chained assignment)
            df_clean[col] = df_clean[col].fillna(fillval)
        else:
            if text_strategy == "mode":
                try:
                    fillval = df_clean[col].mode().iat[0]
                except Exception:
                    fillval = ""
            else:
                fillval = text_constant
            df_clean[col] = df_clean[col].fillna(fillval)

    st.success("Imputation applied to preview `df_clean` (not yet saved).")
    st.dataframe(df_clean.head(10), height=240)

# button to create cleaned copy permanently (overwrite df_clean variable)
if st.button("Create cleaned copy (overwrite preview)"):
    # df_clean already prepared above if apply_impute clicked; else copy original
    if not apply_impute:
        df_clean = df.copy()
    st.success("Cleaned copy created.")
    st.dataframe(df_clean.head(10), height=240)

# download cleaned CSV
def convert_df_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")

if st.button("Download cleaned CSV"):
    csv_bytes = convert_df_to_csv_bytes(df_clean)
    st.download_button(label="Download cleaned.csv", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")

# ---- Missing-values visualization ----
st.subheader("Missing values overview")
missing_counts = df.isnull().sum()
if missing_counts.sum() == 0:
    st.info("No missing values found.")
else:
    st.bar_chart(missing_counts[missing_counts > 0])

# ---- Visualizations ----
st.subheader("📈 Visualizations")
vis_col = st.selectbox("Select numeric column to plot", options=num_cols + ["(none)"])
if vis_col and vis_col != "(none)":
    st.write("Histogram and boxplot for:", vis_col)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df_clean[vis_col].dropna(), bins=30)
    axes[0].set_title(f"Histogram: {vis_col}")
    axes[1].boxplot(df_clean[vis_col].dropna(), vert=False)
    axes[1].set_title(f"Boxplot: {vis_col}")
    st.pyplot(fig)

# ---- Optional ML: simple Logistic Regression ----
st.subheader("🔎 Quick ML: Train simple LogisticRegression (if dataset has binary target 'Outcome')")

if 'Outcome' in df_clean.columns:
    st.write("Found `Outcome` column — ready to train a simple model.")
    test_size = st.slider("Test size (%)", 10, 50, 20)
    do_train = st.button("Train LogisticRegression")
    if do_train:
        # prepare
        X = df_clean.drop(columns=['Outcome'])
        y = df_clean['Outcome']
        # drop non-numeric columns or encode simply
        X_numeric = X.select_dtypes(include=[np.number]).copy()
        if X_numeric.shape[1] == 0:
            st.error("No numeric features available for simple training.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_numeric, y, test_size=test_size/100, random_state=42, stratify=y if len(np.unique(y))>1 else None
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_s, y_train)

            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:,1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.3f}")
            if probs is not None and len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, probs)
                st.success(f"ROC AUC: {auc:.3f}")
            st.text("Classification report:")
            st.text(classification_report(y_test, preds))
else:
    st.info("No `Outcome` column found. To use Quick ML, add a binary target column named `Outcome`.")

# ---- Footer ----
st.markdown("---")
st.caption("Built with ❤️ — VizClean AI (local). Changes: fixed pandas chained-assignment, added cleaned download & quick ML.")
