# streamlit_app.py
"""
VizClean AI - Streamlit
Full app with:
- CSV upload
- Date auto-detection & parsing (improved, no infer_datetime_format)
- Missing value imputation + recommendations + one-click apply
- Correlation (compact top-k) with Plotly/Matplotlib toggle
- Missing matrix & missing bars (Plotly/Matplotlib toggle)
- Forecast visualization (if ds/yhat present) with Plotly/Matplotlib toggle
- Export cleaned CSV/Excel(zip)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import zipfile
import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="VizClean AI - Streamlit", layout="wide")


# -------------------------
# Helpers
# -------------------------
def df_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    d = []
    for c in cols:
        non_null = int(df[c].notna().sum())
        dtype = str(df[c].dtype)
        missing = int(df[c].isna().sum())
        unique = int(df[c].nunique(dropna=True))
        d.append({"column": c, "dtype": dtype, "non_null": non_null, "missing": missing, "unique": unique})
    return pd.DataFrame(d)


def df_auto_insights(df: pd.DataFrame, max_rows=5):
    insights = []
    insights.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    missing_total = int(df.isna().sum().sum())
    insights.append(f"Total missing values: {missing_total}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        insights.append("Numeric summary (first 5 cols shown):")
        insights.append(desc.head(max_rows).to_string())
    return "\n\n".join(insights)


# ---- Improved date detection & parsing (no infer_datetime_format) ----
def detect_and_parse_date_columns(df: pd.DataFrame, sample_size: int = 500, min_fraction: float = 0.6):
    """
    Detect likely date-like columns and parse them to datetime.

    Strategy:
      - For columns whose name suggests a date ('date','day','time','ds') we attempt parsing directly.
      - For object columns we sample up to `sample_size` non-null values and try parsing them with
        several strategies (default, dayfirst=True, yearfirst=True). We pick the strategy that
        successfully parses the largest fraction of the sample, and if that fraction >= min_fraction,
        we parse the whole column with that strategy.
      - Returns (new_df, parsed_columns_list).
    """
    import pandas as _pd
    from dateutil.parser import parse as _dateutil_parse  # fallback if needed

    df = df.copy()
    parsed_any = []

    def _try_parse_series(series, dayfirst=False, yearfirst=False):
        try:
            parsed = _pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, yearfirst=yearfirst)
            non_na = parsed.notna().sum()
            total = max(1, series.dropna().shape[0])
            frac = non_na / total
            return parsed, frac
        except Exception:
            # fallback: elementwise sampling via dateutil
            sample_vals = series.dropna().astype(str).head(sample_size)
            success = 0
            total = 0
            for v in sample_vals:
                total += 1
                try:
                    _ = _dateutil_parse(v, dayfirst=dayfirst, yearfirst=yearfirst)
                    success += 1
                except Exception:
                    pass
            frac = success / max(1, total) if total > 0 else 0.0
            return None, frac

    for col in df.columns:
        name = col.lower()
        try:
            # Strong signal columns
            if 'date' in name or 'day' in name or 'time' in name or col.lower() == 'ds':
                parsed, frac = _try_parse_series(df[col], dayfirst=False, yearfirst=False)
                if frac < min_fraction:
                    parsed_df, frac_df = _try_parse_series(df[col], dayfirst=True, yearfirst=False)
                    parsed_yf, frac_yf = _try_parse_series(df[col], dayfirst=False, yearfirst=True)
                    best_parsed, best_frac = parsed, frac
                    if parsed_df is not None and frac_df > best_frac:
                        best_parsed, best_frac = parsed_df, frac_df
                    if parsed_yf is not None and frac_yf > best_frac:
                        best_parsed, best_frac = parsed_yf, frac_yf
                    if best_frac >= min_fraction and best_parsed is not None:
                        df[col] = best_parsed
                        parsed_any.append(col)
                    else:
                        if parsed is not None:
                            df[col] = parsed
                            parsed_any.append(col)
                else:
                    df[col] = parsed
                    parsed_any.append(col)
                continue

            # For object or string-like columns try sampling
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                sample = df[col].dropna().astype(str).head(sample_size)
                if sample.empty:
                    continue
                # try variants
                results = []
                for dayfirst, yearfirst in [(False, False), (True, False), (False, True)]:
                    _, frac = _try_parse_series(sample, dayfirst=dayfirst, yearfirst=yearfirst)
                    results.append(((dayfirst, yearfirst), frac))
                (best_dayfirst, best_yearfirst), best_frac = max(results, key=lambda x: x[1])
                if best_frac >= min_fraction:
                    parsed_full, _ = _try_parse_series(df[col], dayfirst=best_dayfirst, yearfirst=best_yearfirst)
                    if parsed_full is not None and parsed_full.notna().sum() > 0:
                        df[col] = parsed_full
                        parsed_any.append(col)
        except Exception:
            # ignore and continue
            continue

    return df, parsed_any


def compute_fill_recommendations(df: pd.DataFrame):
    """Recommend mean/median per numeric column based on skewness and outlier ratio."""
    recs = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            recs.append((c, "no-data", "no non-missing values"))
            continue
        skew = float(s.skew())
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            outlier_ratio = 0.0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_ratio = float(((s < lower) | (s > upper)).mean())
        if outlier_ratio > 0.05 or abs(skew) > 1:
            suggestion = "median"
            reason = f"outlier_ratio={outlier_ratio:.2%}, skew={skew:.2f}"
        else:
            suggestion = "mean"
            reason = f"outlier_ratio={outlier_ratio:.2%}, skew={skew:.2f}"
        recs.append((c, suggestion, reason))
    return pd.DataFrame(recs, columns=["column", "suggested_strategy", "reason"])


def impute_dataframe(df: pd.DataFrame, numeric_strategy='mean', text_strategy='mode', columns=None):
    df = df.copy()
    if columns is None or len(columns) == 0:
        columns = df.columns.tolist()

    datetime_cols = [c for c in columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in columns if (c not in datetime_cols and not pd.api.types.is_numeric_dtype(df[c]))]

    # Datetime: forward/backfill
    for c in datetime_cols:
        if df[c].isna().any():
            try:
                df[c] = df[c].fillna(method='ffill').fillna(method='bfill')
            except Exception:
                df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else pd.NaT)

    # Numeric
    for c in numeric_cols:
        if df[c].isna().any():
            if numeric_strategy == 'mean':
                val = df[c].mean()
            elif numeric_strategy == 'median':
                val = df[c].median()
            else:
                val = 0
            df[c].fillna(val, inplace=True)

    # Text
    for c in text_cols:
        if df[c].isna().any():
            if text_strategy == 'mode':
                mode_series = df[c].mode(dropna=True)
                val = mode_series.iloc[0] if not mode_series.empty else ""
            elif text_strategy == 'empty':
                val = ""
            else:
                val = ""
            df[c].fillna(val, inplace=True)

    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='cleaned')
    output.seek(0)
    return output.getvalue()


def make_zip_with_excel(df: pd.DataFrame, excel_name="cleaned.xlsx") -> bytes:
    excel_bytes = to_excel_bytes(df)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(excel_name, excel_bytes)
    buf.seek(0)
    return buf.getvalue()


# -------------------------
# UI
# -------------------------
st.title("✨ VizClean AI — Streamlit")
st.write("Transform raw data into clean insights in seconds.")

# top controls: use plotly toggle + global apply recommended imputation
top_left, top_right = st.columns([3, 1])
with top_right:
    use_plotly = st.checkbox("Use interactive Plotly charts", value=True)
with top_left:
    st.markdown(" ")

upload_col, actions_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

        if df is not None:
            df_parsed, parsed_cols = detect_and_parse_date_columns(df)
            if parsed_cols:
                st.success(f"Detected & parsed date columns: {parsed_cols}")
            st.session_state['uploaded_df'] = df_parsed
            st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({df_parsed.shape[0]} rows, {df_parsed.shape[1]} columns)")
    else:
        df = st.session_state.get('uploaded_df', None)

with actions_col:
    st.markdown("### Actions")
    if 'uploaded_df' in st.session_state:
        if st.button("Reset uploaded data"):
            st.session_state.pop('uploaded_df', None)
            st.session_state.pop('cleaned_df', None)
            st.experimental_rerun()
    else:
        st.info("Upload a CSV to begin")

st.markdown("---")

if 'uploaded_df' in st.session_state:
    df = st.session_state['uploaded_df']
    left, right = st.columns([2, 1])
    with left:
        st.header("Original Data")
        st.write(f"First 10 rows — ({df.shape[0]} rows total)")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("#### Dataset Summary")
        summary_df = df_summary(df)
        st.dataframe(summary_df, use_container_width=True)
    with right:
        st.header("Auto Insights")
        st.code(df_auto_insights(df), language=None)

    st.markdown("---")

    # Fill Missing UI + auto-recommend apply
    st.subheader("🧽 Fill Missing Values (Imputation)")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        enabled = st.checkbox("Enable Fill Missing Values", value=False)
    with col2:
        numeric_strategy = st.selectbox("Numeric strategy", ["mean", "median"], index=0)
    with col3:
        text_strategy = st.selectbox("Text strategy", ["mode", "empty"], index=0)

    all_cols = df.columns.tolist()
    cols_to_include = st.multiselect("Columns to impute (leave empty = all columns)", options=all_cols)

    st.markdown("#### Auto-fill strategy suggestions (based on skewness & outliers)")
    try:
        recs_df = compute_fill_recommendations(df)
        st.dataframe(recs_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute auto-fill suggestions: {e}")
        recs_df = pd.DataFrame(columns=["column", "suggested_strategy", "reason"])

    # One-click apply recommended imputations
    if st.button("Apply recommended imputation (per-column)"):
        try:
            map_strat = dict(zip(recs_df['column'], recs_df['suggested_strategy']))
            cleaned = df.copy()
            for col, strat in map_strat.items():
                if col in cleaned.columns and cleaned[col].isna().any():
                    if pd.api.types.is_numeric_dtype(cleaned[col]):
                        if strat == 'median':
                            cleaned[col].fillna(cleaned[col].median(), inplace=True)
                        else:
                            cleaned[col].fillna(cleaned[col].mean(), inplace=True)
                    else:
                        if strat == 'mode':
                            mode_series = cleaned[col].mode(dropna=True)
                            val = mode_series.iloc[0] if not mode_series.empty else ""
                            cleaned[col].fillna(val, inplace=True)
                        else:
                            cleaned[col].fillna("", inplace=True)
            st.session_state['cleaned_df'] = cleaned
            st.success("✅ Applied recommended imputation and updated cleaned_df")
        except Exception as e:
            st.error(f"Could not apply recommended imputation: {e}")

    if enabled:
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        text_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
        datetime_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        st.write(f"Numeric columns: {numeric_cols}")
        st.write(f"Text columns: {text_cols}")
        st.write(f"Datetime columns: {datetime_cols}")

        if st.button("Preview imputed (first 5 rows)"):
            preview_cols = cols_to_include if cols_to_include else None
            try:
                preview_df = impute_dataframe(df, numeric_strategy=numeric_strategy, text_strategy=text_strategy, columns=preview_cols)
                st.write("Preview (first 5 rows after imputation):")
                st.dataframe(preview_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Could not create preview: {e}")

        if st.button("Apply imputation & replace cleaned data"):
            try:
                cleaned = impute_dataframe(df, numeric_strategy=numeric_strategy, text_strategy=text_strategy, columns=(cols_to_include if cols_to_include else None))
                st.session_state['cleaned_df'] = cleaned
                st.success("✅ Missing values filled and cleaned_df updated")
                st.dataframe(cleaned.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Could not apply imputation: {e}")
    else:
        st.info("Enable the checkbox to choose and apply imputation strategies.")

    st.markdown("---")

    # Correlation
    st.subheader("📈 Correlation (numeric columns) — compact/top-k")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        try:
            corr_full = numeric_df.corr()
            if corr_full.isnull().all().all():
                corr_full = numeric_df.dropna().corr()
            if corr_full.isnull().all().all():
                st.info("Correlation could not be computed (not enough complete pairs).")
            else:
                st.write("Correlation table (rounded):")
                st.dataframe(corr_full.round(3), use_container_width=True)

                k = st.slider("Show top k correlated columns for compact heatmap", min_value=3, max_value=min(20, numeric_df.shape[1]), value=8)
                abs_corr = corr_full.abs().where(~np.eye(corr_full.shape[0], dtype=bool))
                pairs = abs_corr.unstack().dropna().sort_values(ascending=False)
                seen = set(); unique_pairs = []
                for (a, b), val in pairs.items():
                    key = tuple(sorted((a, b)))
                    if key in seen:
                        continue
                    seen.add(key); unique_pairs.append((a, b))
                top_pairs = unique_pairs[:k]
                cols = []
                for a, b in top_pairs:
                    if a not in cols: cols.append(a)
                    if b not in cols: cols.append(b)
                cols = cols[:k]
                corr_sub = corr_full.loc[cols, cols]

                if use_plotly:
                    fig = px.imshow(corr_sub.values,
                                    x=corr_sub.columns,
                                    y=corr_sub.index,
                                    color_continuous_scale='RdBu',
                                    zmin=-1, zmax=1,
                                    aspect='auto',
                                    text_auto='.2f')
                    fig.update_layout(title="Compact correlation heatmap (interactive)", height=max(300, 40 * len(cols)))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(cols)), max(4, 0.6 * len(cols))))
                    sns.heatmap(corr_sub, annot=True, fmt=".2f", linewidths=0.5, ax=ax, cmap="vlag", center=0, cbar_kws={"shrink": 0.6})
                    ax.set_title("Compact correlation heatmap (top correlated columns)")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
        except Exception as e:
            st.error(f"Could not compute/render correlation: {e}")
    else:
        st.info("Not enough numeric columns to compute correlation.")

    st.markdown("---")

    # Missing-values matrix
    st.subheader("🕳️ Missing-values matrix (sample / full)")
    miss_counts = df.isna().sum().sort_values(ascending=False)
    miss_counts_df = miss_counts.reset_index(); miss_counts_df.columns = ['column', 'missing']
    st.write("Missing counts per column:"); st.dataframe(miss_counts_df, use_container_width=True)

    total_missing = int(miss_counts.sum())
    missing_row_indices = df.index[df.isna().any(axis=1)].tolist()
    default_show_last = False; default_sample_rows = 0
    if total_missing > 0 and len(missing_row_indices) > 0:
        frac_last = sum(1 for i in missing_row_indices if i >= 0.9 * len(df)) / len(missing_row_indices)
        if frac_last > 0.5:
            default_show_last = True
            default_sample_rows = min(1000, max(50, int(0.05 * len(df))))

    max_cols_plot = int(st.number_input("Max columns to show in missing-matrix heatmap", min_value=1, max_value=len(df.columns), value=min(6, len(df.columns))))
    sample_rows = int(st.number_input("Max rows to visualize (0 = all rows, otherwise sample first N)", min_value=0, max_value=100000, value=default_sample_rows))
    show_last = st.checkbox("Show last rows instead of first (useful when missing at dataset end)", value=default_show_last)

    miss_bool = df.isna()
    cap = 5000
    if sample_rows and sample_rows > 0:
        miss_plot_df = miss_bool.tail(sample_rows) if show_last else miss_bool.head(sample_rows)
    else:
        miss_plot_df = miss_bool.tail(min(cap, miss_bool.shape[0])) if show_last else (miss_bool if miss_bool.shape[0] <= cap else miss_bool.head(cap))

    miss_plot_df = miss_plot_df.iloc[:, :max_cols_plot]

    if miss_plot_df.values.sum() == 0:
        st.info("Selected window contains no missing values to visualize. Try checking 'Show last rows' or increasing 'Max rows to visualize'.")
    else:
        if use_plotly:
            z = miss_plot_df.T.astype(int).values
            fig = px.imshow(z, x=[str(i) for i in range(miss_plot_df.shape[0])], y=miss_plot_df.columns,
                            color_continuous_scale=["white", "black"], aspect='auto')
            fig.update_layout(title="Missing-values matrix (black=missing)", coloraxis_showscale=False, height=max(300, 20 * len(miss_plot_df.columns)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            try:
                fig2, ax2 = plt.subplots(figsize=(min(20, 0.5 * miss_plot_df.shape[1] + 3), min(8, 0.02 * miss_plot_df.shape[0] + 3)))
                sns.heatmap(miss_plot_df.T, cbar=False, cmap='gray_r', ax=ax2)
                ax2.set_yticks(np.arange(len(miss_plot_df.columns)) + 0.5)
                ax2.set_yticklabels(miss_plot_df.columns, fontsize=8, rotation=0)
                ax2.set_xticks([])
                ax2.set_title("Missing-values matrix (dark = missing)")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.error(f"Could not render missing-values matrix: {e}")

    st.markdown("---")

    # Cleaned Data & exports
    st.subheader("Cleaned Data")
    cleaned_df = st.session_state.get('cleaned_df', None)
    if cleaned_df is None:
        st.warning("No cleaned dataset yet. Apply imputation or press the 'Create cleaned copy' button to copy original into cleaned.")
        if st.button("Create cleaned copy from original"):
            st.session_state['cleaned_df'] = df.copy()
            st.success("Created cleaned_df from original.")
    else:
        st.write(f"Cleaned dataset preview ({cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns)")
        st.dataframe(cleaned_df.head(10), use_container_width=True)
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            try:
                csv_bytes = to_csv_bytes(cleaned_df)
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_name = f"cleaned_{now}.csv"
                st.download_button("Download cleaned CSV", data=csv_bytes, file_name=csv_name, mime="text/csv")
            except Exception as e:
                st.error(f"Could not prepare CSV download: {e}")
        with exp_col2:
            try:
                zip_bytes = make_zip_with_excel(cleaned_df, excel_name=f"cleaned_{now}.xlsx")
                zip_name = f"cleaned_{now}.zip"
                st.download_button("Download cleaned Excel in ZIP", data=zip_bytes, file_name=zip_name, mime="application/zip")
            except Exception as e:
                st.error(f"Could not prepare Excel ZIP download: {e}")

    st.markdown("---")

    # Missing bar chart
    st.subheader("Missing values per column (bar chart)")
    miss_counts = df.isna().sum().sort_values(ascending=False)
    if miss_counts.sum() > 0:
        if use_plotly:
            fig = px.bar(x=miss_counts.values, y=miss_counts.index, orientation='h', labels={'x': 'Missing count', 'y': 'column'})
            fig.update_layout(height=max(300, 20 * len(miss_counts)), title="Missing counts per column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            try:
                fig3, ax3 = plt.subplots(figsize=(8, max(3, 0.25 * len(miss_counts))))
                sns.barplot(x=miss_counts.values, y=miss_counts.index, ax=ax3)
                ax3.set_xlabel("Missing count")
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
            except Exception as e:
                st.error(f"Could not render missing-values bar chart: {e}")
    else:
        st.info("No missing values found in the dataset.")

    st.markdown("---")

    # Forecast visualization & resampling
    st.subheader("📊 Forecast visualization & resampling")
    if 'ds' in df.columns and 'yhat' in df.columns:
        ts_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(ts_df['ds']):
            ts_df['ds'] = pd.to_datetime(ts_df['ds'], errors='coerce')

        st.write("Choose date range and aggregation for visualization:")
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            min_date = ts_df['ds'].min()
            max_date = ts_df['ds'].max()
            try:
                date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
            except Exception:
                date_range = st.date_input("Date range", value=(pd.to_datetime(min_date).date(), pd.to_datetime(max_date).date()))
        with col_b:
            agg = st.selectbox("Resample / aggregate", ["none", "D", "W", "M", "Q", "Y"], index=0,
                               help="D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly; 'none' = no resample")
        with col_c:
            rolling_win = st.number_input("Rolling window (periods) for smoothing (0 = none)", min_value=0, max_value=365, value=0)

        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        vis_df = ts_df[(ts_df['ds'] >= start_dt) & (ts_df['ds'] <= end_dt)].set_index('ds').sort_index()

        numeric_series = [c for c in vis_df.columns if pd.api.types.is_numeric_dtype(vis_df[c])]
        default_plot = ['yhat', 'yhat_lower', 'yhat_upper'] if 'yhat' in vis_df.columns else numeric_series[:3]
        plot_series = st.multiselect("Series to plot", options=numeric_series, default=default_plot)
        if len(plot_series) == 0:
            st.info("Choose at least one series to plot.")
        else:
            plot_df = vis_df[plot_series].copy()
            if agg != "none":
                try:
                    plot_df = plot_df.resample(agg).mean()
                except Exception:
                    st.warning("Could not resample with the chosen period; showing raw series.")
            if rolling_win and rolling_win > 0:
                plot_df = plot_df.rolling(window=rolling_win, min_periods=1, center=True).mean()

            if plot_df.empty:
                st.info("No data in the selected date-range.")
            else:
                try:
                    if use_plotly:
                        fig = go.Figure()
                        if 'yhat' in plot_df.columns:
                            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['yhat'], name='yhat', mode='lines'))
                        if 'y' in plot_df.columns:
                            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['y'], name='y (observed)', mode='markers', marker=dict(size=6, opacity=0.6)))
                        if 'yhat_lower' in plot_df.columns and 'yhat_upper' in plot_df.columns:
                            fig.add_trace(go.Scatter(x=list(plot_df.index) + list(plot_df.index[::-1]),
                                                     y=list(plot_df['yhat_upper']) + list(plot_df['yhat_lower'][::-1]),
                                                     fill='toself', fillcolor='rgba(173,216,230,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=True, name='yhat interval'))
                        for colname in plot_df.columns:
                            if colname in ['yhat', 'yhat_lower', 'yhat_upper', 'y']:
                                continue
                            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[colname], name=colname, mode='lines'))
                        fig.update_layout(title="Forecast visualization (interactive)", xaxis_title="Date", yaxis_title="Value", height=450)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig4, ax4 = plt.subplots(figsize=(10, 4))
                        if 'yhat' in plot_df.columns:
                            ax4.plot(plot_df.index, plot_df['yhat'], label='yhat', linewidth=1.8)
                        if 'y' in plot_df.columns:
                            ax4.scatter(plot_df.index, plot_df['y'], label='y (observed)', s=12, alpha=0.6)
                        if 'yhat_lower' in plot_df.columns and 'yhat_upper' in plot_df.columns:
                            ax4.fill_between(plot_df.index, plot_df['yhat_lower'], plot_df['yhat_upper'], alpha=0.2, label='yhat interval')
                        for colname in plot_df.columns:
                            if colname in ['yhat', 'yhat_lower', 'yhat_upper', 'y']:
                                continue
                            ax4.plot(plot_df.index, plot_df[colname], label=colname, linewidth=1.0, alpha=0.9)
                        ax4.set_title("Forecast visualization")
                        ax4.set_xlabel("Date")
                        ax4.set_ylabel("Value")
                        ax4.legend(loc='upper left', fontsize='small', ncol=2)
                        fig4.autofmt_xdate()
                        fig4.tight_layout()
                        st.pyplot(fig4)
                        plt.close(fig4)

                    # downloads: CSV and PNG
                    csv_bytes = plot_df.reset_index().to_csv(index=False).encode('utf-8')
                    st.download_button("Download plotted data (CSV)", data=csv_bytes, file_name="forecast_plot_data.csv", mime="text/csv")

                    if use_plotly:
                        try:
                            img_bytes = fig.to_image(format="png", scale=2)
                            st.download_button("Download plot (PNG)", data=img_bytes, file_name="forecast_plot.png", mime="image/png")
                        except Exception:
                            buf = BytesIO()
                            fig_mat, ax_mat = plt.subplots(figsize=(10, 4))
                            for col in plot_df.columns:
                                ax_mat.plot(plot_df.index, plot_df[col], label=col)
                            ax_mat.legend()
                            fig_mat.tight_layout()
                            fig_mat.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            st.download_button("Download plot (PNG)", data=buf, file_name="forecast_plot.png", mime="image/png")
                            plt.close(fig_mat)
                    else:
                        buf = BytesIO()
                        fig4.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button("Download plot (PNG)", data=buf, file_name="forecast_plot.png", mime="image/png")

                    try:
                        stats = plot_df.agg(['mean', 'std', 'min', 'max']).T
                        st.write("Summary stats of plotted series (mean/std/min/max):")
                        st.dataframe(stats.round(4), use_container_width=True)
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"Could not render forecast plot: {e}")
    else:
        st.info("No 'ds' (date) and 'yhat' columns found for time-series visualization. Upload a forecast-like CSV with 'ds' and 'yhat' columns.")

    st.markdown("---")

    # Profiling / AI-style summary
    st.subheader("🔎 Data Profiling & AI-style summary")
    prof_col1, prof_col2 = st.columns([2, 1])
    with prof_col2:
        want_profile = st.button("Generate full profile (HTML)")
    with prof_col1:
        st.write("Quick local summary:")
        st.write(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write(f"- Total missing values: {int(df.isna().sum().sum())}")
        try:
            st.write(f"- Numeric columns: {list(numeric_df.columns)}")
        except Exception:
            pass
        st.write("Top 5 columns by missing values:")
        try:
            st.write(miss_counts.head(5))
        except Exception:
            pass
        try:
            st.write("Auto-fill suggestions summary (see table above):")
            st.dataframe(recs_df.set_index("column")[["suggested_strategy"]], use_container_width=True)
        except Exception:
            pass

    if want_profile:
        try:
            from ydata_profiling import ProfileReport
            profile = ProfileReport(df, title="VizClean AI - Profile", explorative=True)
            html = profile.to_html()
            b = html.encode('utf-8')
            st.download_button("Download profile HTML", data=b, file_name=f"profile_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html")
            st.success("Profile generated — download available.")
        except Exception as e:
            st.error(f"Could not generate profile. Make sure 'ydata-profiling' (ydata_profiling) is installed. Error: {e}")

    st.markdown("---")

    # Rows with missing (context)
    st.subheader("Rows that contain any missing values (first 100 rows)")
    rows_with_missing = df[df.isna().any(axis=1)]
    if rows_with_missing.shape[0] == 0:
        st.info("No rows with missing values.")
    else:
        st.dataframe(rows_with_missing.head(100), use_container_width=True)
    st.write(f"Rows with missing: {rows_with_missing.shape[0]} / {df.shape[0]}")

else:
    st.write("Waiting for CSV upload...")

# Footer
st.markdown("---")
st.write("Built with ❤️ — VizClean AI (Streamlit)")
