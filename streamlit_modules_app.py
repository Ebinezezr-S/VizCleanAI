# streamlit_modules_app.py
# VizClean — Single-file multipage app (full, updated)
import io
import json
import os
import re
from datetime import datetime

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# -------------------------
# Config / paths
# -------------------------
st.set_page_config(page_title="VizClean — Industry Demos", layout="wide")
ROOT = os.getcwd()
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
CONSENT_LOG_PATH = os.path.join(MODEL_DIR, "consent_audit.log")


# -------------------------
# Helper: file readers
# -------------------------
def read_uploaded_file_bytes(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".pdf"):
        # PDF table extraction optional: requires tabula-py & java; fallback: error
        try:
            import tabula

            dfs = tabula.read_pdf(uploaded_file, pages="all", multiple_tables=True)
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            raise RuntimeError("No tables found in PDF via tabula.")
        except Exception as e:
            raise RuntimeError("PDF extraction unavailable: " + str(e))
    raise RuntimeError("Unsupported file type: " + uploaded_file.name)


# -------------------------
# Consent + PII detector
# -------------------------
PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone_international": re.compile(r"\+?\d[\d\s\-]{6,}\d"),
    "aadhar_like": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
}


def log_consent_action(action: str, details: dict = None):
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "action": action,
        "details": details or {},
    }
    try:
        with open(CONSENT_LOG_PATH, "a", encoding="utf8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def detect_pii_in_df(df: pd.DataFrame, max_preview=5):
    results = {}
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(300)
        matches = []
        for val in sample:
            for name, patt in PII_PATTERNS.items():
                if patt.search(val):
                    matches.append({"pattern": name, "value": val})
                    break
        if matches:
            uniq_vals = []
            for m in matches:
                if m["value"] not in uniq_vals:
                    uniq_vals.append(m["value"])
                if len(uniq_vals) >= max_preview:
                    break
            results[col] = {"count_examples": len(uniq_vals), "examples": uniq_vals}
    return results


# -------------------------
# Auto-detect field roles + summary
# -------------------------
def detect_field_roles(df: pd.DataFrame, max_unique_for_categorical=50):
    roles = {}
    for col in df.columns:
        ser = df[col]
        dtype = ser.dtype
        try:
            nunique = int(ser.nunique(dropna=True))
        except Exception:
            nunique = -1
        pct_null = ser.isna().mean()
        info = {
            "dtype": str(dtype),
            "nunique": int(nunique) if nunique >= 0 else None,
            "pct_null": float(pct_null),
        }
        if re.search(r"(id$|_id$|^id$|identifier)", col, re.I) or (
            np.issubdtype(dtype, np.integer)
            and nunique > 50
            and nunique == len(ser.dropna())
        ):
            info["role"] = "id"
        elif np.issubdtype(dtype, np.number):
            info["role"] = "numeric"
        elif np.issubdtype(dtype, np.datetime64) or re.search(
            r"(date|time|timestamp|year)", col, re.I
        ):
            info["role"] = "datetime"
        elif nunique != -1 and nunique <= max_unique_for_categorical:
            info["role"] = "categorical"
        else:
            info["role"] = "text"
        info["likely_target"] = col.lower() in (
            "target",
            "label",
            "outcome",
            "y",
            "class",
            "status",
            "fraud",
            "is_fraud",
            "churn",
        )
        roles[col] = info
    return roles


def quick_dataset_summary(df: pd.DataFrame, roles=None, max_preview_rows=5):
    nrows, ncols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_pct = df.isna().mean().mean() * 100
    top_numeric = sorted(
        numeric_cols, key=lambda c: df[c].var() if c in df.columns else 0, reverse=True
    )[:3]
    candidate_targets = []
    if roles:
        candidate_targets = [c for c, i in roles.items() if i.get("likely_target")]
    if not candidate_targets:
        for c in df.columns:
            vals = df[c].dropna().unique()
            if len(vals) <= 3 and set(map(str, vals)) <= set(
                ["0", "1", "yes", "no", "true", "false"]
            ):
                candidate_targets.append(c)
    sentences = []
    sentences.append(f"The dataset has {nrows} rows and {ncols} columns.")
    if candidate_targets:
        sentences.append(f"Possible target columns: {', '.join(candidate_targets)}.")
    else:
        sentences.append("No obvious target column detected automatically.")
    if numeric_cols:
        sentences.append(
            f"Numeric cols: {', '.join(numeric_cols[:5])}{('...' if len(numeric_cols)>5 else '')}."
        )
    if missing_pct > 0:
        sentences.append(
            f"Average missingness: {missing_pct:.1f}% — consider cleaning/imputing."
        )
    if top_numeric:
        sentences.append(f"High-variance numeric cols: {', '.join(top_numeric)}.")
    sentences.append("Preview the first rows to validate column roles.")
    preview = df.head(max_preview_rows)
    return {
        "summary_text": " ".join(sentences),
        "preview_df": preview,
        "candidate_targets": candidate_targets,
    }


# -------------------------
# Simple models (healthcare, finance, recommender, transport, manufacturing)
# -------------------------
HEALTH_MODEL_PATH = os.path.join(MODEL_DIR, "healthcare_disease_model.joblib")


def hc_train_and_save(df: pd.DataFrame, target_col: str = "target"):
    # Very small safety checks: keep numeric-only X for logistic regression
    df = df.dropna()
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Keep numeric features only for this simple demo
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[0] < 10:
        raise ValueError("Not enough rows (need >=10)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = None
    joblib.dump(model, HEALTH_MODEL_PATH)
    return {
        "accuracy": float(acc),
        "roc_auc": (float(auc) if auc is not None else None),
    }


def hc_load_model():
    try:
        return joblib.load(HEALTH_MODEL_PATH)
    except Exception:
        return None


def hc_predict(df: pd.DataFrame):
    model = hc_load_model()
    if model is None:
        raise RuntimeError("Health model not trained yet.")
    Xnum = df.select_dtypes(include=[np.number])
    if Xnum.shape[1] == 0:
        raise ValueError("No numeric features available for prediction.")
    return model.predict(Xnum)


FIN_MODEL_PATH = os.path.join(MODEL_DIR, "finance_fraud_if.joblib")


def fin_train_and_save(df: pd.DataFrame, features=None, contamination=0.01):
    df = df.dropna()
    if features is None:
        features = df.columns.tolist()
    X = df[features]
    model = IsolationForest(
        n_estimators=100, contamination=contamination, random_state=42
    )
    model.fit(X)
    joblib.dump((model, features), FIN_MODEL_PATH)
    return {"status": "saved", "n_samples": len(X), "features": features}


def fin_load_model():
    try:
        return joblib.load(FIN_MODEL_PATH)
    except Exception:
        return None


def fin_score_transactions(df: pd.DataFrame):
    loaded = fin_load_model()
    if loaded is None:
        raise RuntimeError("Finance model not trained yet.")
    model, features = loaded
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[features]
    scores = model.decision_function(X)
    is_anom = model.predict(X) == -1
    return pd.DataFrame({"score": scores, "is_fraud": is_anom})


# Recommender helpers
def rec_recommend_items_from_user_item_matrix(
    user_item_df: pd.DataFrame, user_id, top_k=10
):
    if user_id not in user_item_df.index:
        raise KeyError("User not in matrix")
    user_vec = user_item_df.loc[user_id].values.reshape(1, -1)
    sim = cosine_similarity(user_vec, user_item_df.fillna(0))
    sim_scores = pd.Series(sim.flatten(), index=user_item_df.index)
    weights = sim_scores
    scores = (weights.values.reshape(-1, 1) * user_item_df.fillna(0).values).sum(axis=0)
    scores_series = pd.Series(scores, index=user_item_df.columns)
    return scores_series.sort_values(ascending=False).head(top_k)


def rec_recommend_similar_items_from_item_features(
    item_features_df: pd.DataFrame, item_id, top_k=10
):
    if item_id not in item_features_df.index:
        raise KeyError("Item not in features")
    mat = item_features_df.fillna(0).values
    sim = cosine_similarity(mat)
    idx = item_features_df.index.get_loc(item_id)
    scores = pd.Series(sim[idx], index=item_features_df.index)
    return scores.sort_values(ascending=False).drop(item_id).head(top_k)


# Transport graph
def tr_build_graph_from_edges(edges_df: pd.DataFrame):
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        w = float(row.get("weight", 1.0)) if "weight" in row.index else 1.0
        G.add_edge(str(row["source"]), str(row["target"]), weight=w)
    return G


def tr_shortest_route(G: nx.DiGraph, origin, destination):
    origin = str(origin)
    destination = str(destination)
    if origin not in G.nodes:
        raise KeyError(f"Origin '{origin}' not in graph")
    if destination not in G.nodes:
        raise KeyError(f"Destination '{destination}' not in graph")
    path = nx.shortest_path(G, source=origin, target=destination, weight="weight")
    length = nx.shortest_path_length(
        G, source=origin, target=destination, weight="weight"
    )
    return {"path": path, "length": float(length)}


# Manufacturing PM
MANU_MODEL_PATH = os.path.join(MODEL_DIR, "manufacturing_pm.joblib")


def manu_featurize_sensor_window(df: pd.DataFrame):
    features = {}
    features["mean_all"] = df.mean().mean()
    features["std_all"] = df.std().mean()
    features["max_all"] = df.max().max()
    features["min_all"] = df.min().min()
    features["mean_of_means"] = df.mean().mean()
    return pd.DataFrame([features])


def manu_train_and_save(X_df: pd.DataFrame, y_series: pd.Series):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_df, y_series)
    joblib.dump(model, MANU_MODEL_PATH)
    return {"status": "saved", "n_samples": len(X_df)}


def manu_predict_from_features(X_df: pd.DataFrame):
    try:
        model = joblib.load(MANU_MODEL_PATH)
    except Exception:
        raise RuntimeError("Manufacturing model not trained yet.")
    return model.predict_proba(X_df)[:, 1]


# -------------------------
# Consent & session init
# -------------------------
if "consent_given" not in st.session_state:
    st.session_state["consent_given"] = False
if "consent_timestamp" not in st.session_state:
    st.session_state["consent_timestamp"] = None

CONSENT_TEXT = """
By uploading data you CONSENT to processing in this app for profiling, cleaning, modeling, and temporary storage in the models/ directory. We will NOT share data externally unless you export or explicitly send results.
You can revoke consent at any time.
"""


def consent_widget():
    st.markdown("### Data use consent")
    with st.expander("What you are consenting to"):
        st.write(CONSENT_TEXT)
    col1, col2 = st.columns([3, 1])
    with col1:
        agree = st.checkbox(
            "I have read and I CONSENT to the data usage terms (required to upload)",
            value=st.session_state["consent_given"],
            key="consent_checkbox",
        )
    with col2:
        if st.session_state["consent_given"]:
            if st.button("Revoke consent"):
                st.session_state["consent_given"] = False
                st.session_state["consent_timestamp"] = None
                log_consent_action("revoke", {"reason": "user_clicked_revoke"})
                st.success("Consent revoked.")
        else:
            if st.button("Show consent log"):
                try:
                    if os.path.exists(CONSENT_LOG_PATH):
                        with open(CONSENT_LOG_PATH, "r", encoding="utf8") as f:
                            lines = f.readlines()[-20:]
                        st.code("".join(lines[-10:]))
                    else:
                        st.info("No consent actions logged yet.")
                except Exception as e:
                    st.error("Cannot read consent log: " + str(e))
    if agree and not st.session_state["consent_given"]:
        st.session_state["consent_given"] = True
        st.session_state["consent_timestamp"] = datetime.utcnow().isoformat() + "Z"
        log_consent_action(
            "grant", {"method": "checkbox", "ts": st.session_state["consent_timestamp"]}
        )
        st.success("Consent recorded.")
    elif not agree and st.session_state["consent_given"]:
        st.session_state["consent_given"] = False
        log_consent_action("revoke", {"method": "checkbox_unchecked"})
        st.warning("Consent revoked via checkbox.")


# -------------------------
# Streamlit UI pages
# -------------------------
st.title("VizClean — Industry Demos (All-in-one)")

page = st.sidebar.selectbox(
    "Choose a domain",
    [
        "Overview",
        "Healthcare",
        "Finance",
        "E-commerce",
        "Transport",
        "Manufacturing",
        "Data Inspector",
        "Resources",
    ],
)

# Overview
if page == "Overview":
    st.header("Overview")
    st.markdown(
        """
This demo bundles simple domain modules: Healthcare, Finance, E-commerce, Transport, Manufacturing.
Use the Data Inspector to upload files, auto-detect fields, and get quick summaries.
    """
    )
    st.info(
        "Make sure `models/` is writable. Consent is required before uploading data."
    )

# Healthcare
elif page == "Healthcare":
    st.header("Healthcare — Disease Prediction (example)")
    st.write(
        "Upload CSV with numeric features + 'target' column (0/1) or use sample data."
    )
    consent_widget()
    if not st.session_state["consent_given"]:
        st.info("Give consent to upload and process healthcare data.")
    else:
        uploaded = st.file_uploader(
            "Upload CSV (healthcare)", type=["csv"], key="health_file"
        )
        sample_btn = st.button("Use small sample dataset")
        df = None
        if sample_btn:
            np.random.seed(42)
            df = pd.DataFrame(
                {
                    "age": np.random.randint(20, 80, size=200),
                    "bmi": np.random.normal(27, 5, size=200),
                    "sys_bp": np.random.randint(100, 160, size=200),
                    "chol": np.random.randint(150, 300, size=200),
                }
            )
            df["target"] = (
                0.03 * (df["age"] - 50)
                + 0.02 * (df["bmi"] - 25)
                + np.random.normal(0, 1, len(df))
                > 0
            ).astype(int)
            st.success("Sample data generated (200 rows).")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                # -----------------------
                # convenience: auto-rename common diabetic dataset column 'Outcome' -> 'target'
                # only when 'target' doesn't already exist
                try:
                    if "target" not in df.columns and "Outcome" in df.columns:
                        df = df.rename(columns={"Outcome": "target"})
                        st.info("Auto-rename: 'Outcome' -> 'target' (for convenience).")
                except Exception:
                    pass
                # -----------------------
                st.success("File loaded.")
                log_consent_action(
                    "upload",
                    {
                        "filename": uploaded.name,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    },
                )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df = None
        if df is not None:
            st.subheader("Preview")
            st.dataframe(df.head())
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(f"Numeric columns: {cols}")
            # if there's a sensible candidate target, prefill the text input with it
            roles_preview = detect_field_roles(df)
            candidates = [c for c, i in roles_preview.items() if i.get("likely_target")]
            pre_target = "target"
            if not candidates and "Outcome" in df.columns:
                pre_target = (
                    "target"  # because we auto-rename; otherwise user may enter Outcome
                )
            elif candidates:
                pre_target = candidates[0]
            target_col = st.text_input("Target column name", value=pre_target)
            col1, col2 = st.columns(2)
            # --- improved training button with robust checks ---
            with col1:
                if st.button("Train Model (Healthcare)"):
                    try:
                        if target_col not in df.columns:
                            # give helpful suggestions: case-insensitive match or recommended columns
                            lower_map = {c.lower(): c for c in df.columns}
                            suggestion = None
                            if target_col.lower() in lower_map:
                                suggestion = lower_map[target_col.lower()]
                            candidates2 = [
                                c
                                for c in df.columns
                                if c.lower()
                                in (
                                    "outcome",
                                    "target",
                                    "label",
                                    "y",
                                    "class",
                                    "status",
                                )
                            ]
                            msg = f"Target column '{target_col}' not found. Available columns: {', '.join(df.columns)}."
                            if suggestion:
                                msg += f" Did you mean '{suggestion}'? Try entering that exact name."
                            elif candidates2:
                                msg += f" Candidate target-like columns: {', '.join(candidates2)}."
                            st.error(msg)
                        else:
                            # training uses numeric-only features (safe)
                            with st.spinner("Training logistic regression..."):
                                res = hc_train_and_save(df, target_col=target_col)
                            st.success(
                                f"Trained. Accuracy: {res['accuracy']:.3f}, ROC AUC: {res['roc_auc']}"
                            )
                    except Exception as e:
                        st.error(str(e))
            with col2:
                if st.button("Predict (use features only)"):
                    try:
                        # safe fallback: if target not in columns, use all columns and rely on numeric-select inside hc_predict
                        if target_col in df.columns:
                            X = df.drop(columns=[target_col])
                        else:
                            X = df.copy()
                            st.warning(
                                "Target column not found; attempting prediction using all available columns (numeric-only will be used by the model)."
                            )
                        preds = hc_predict(X)
                        st.write("Predictions (first 20):")
                        st.write(list(preds[:20]))
                    except Exception as e:
                        st.error(str(e))

# Finance
elif page == "Finance":
    st.header("Finance — Fraud Detection (IsolationForest)")
    consent_widget()
    if not st.session_state["consent_given"]:
        st.info("Give consent to upload and process finance data.")
    else:
        uploaded = st.file_uploader(
            "Upload CSV (finance)", type=["csv"], key="fin_file"
        )
        sample_btn = st.button("Use small sample transactions")
        df = None
        if sample_btn:
            rng = np.random.RandomState(42)
            n = 500
            df = pd.DataFrame(
                {
                    "amount": rng.exponential(scale=100, size=n),
                    "age_days": rng.randint(0, 365 * 5, size=n),
                    "num_items": rng.poisson(2, size=n),
                    "hour": rng.randint(0, 24, size=n),
                }
            )
            idx = rng.choice(n, size=10, replace=False)
            df.loc[idx, "amount"] *= 10
            st.success("Sample transactions generated (500 rows).")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success("File loaded.")
                log_consent_action(
                    "upload",
                    {
                        "filename": uploaded.name,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    },
                )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df = None
        if df is not None:
            st.subheader("Preview")
            st.dataframe(df.head())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = st.multiselect(
                "Select features for model", numeric_cols, default=numeric_cols
            )
            contamination = st.slider("Contamination", 0.0, 0.1, 0.01, step=0.001)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Fraud Model"):
                    try:
                        with st.spinner("Training IsolationForest..."):
                            res = fin_train_and_save(
                                df,
                                features=features,
                                contamination=float(contamination),
                            )
                        st.success(f"Model saved. Samples: {res['n_samples']}")
                    except Exception as e:
                        st.error(str(e))
            with col2:
                if st.button("Score Transactions"):
                    try:
                        out = fin_score_transactions(df)
                        st.subheader("Scores (first 20)")
                        st.dataframe(out.head(20))
                        st.table(
                            pd.concat([df.reset_index(drop=True), out], axis=1)
                            .loc[out["is_fraud"]]
                            .head(10)
                        )
                    except Exception as e:
                        st.error(str(e))

# E-commerce
elif page == "E-commerce":
    st.header("E-commerce — Simple Recommender")
    consent_widget()
    if not st.session_state["consent_given"]:
        st.info("Give consent to upload and process ecommerce data.")
    else:
        uploaded = st.file_uploader(
            "Upload CSV (ecommerce)", type=["csv"], key="ecom_file"
        )
        sample_btn = st.button("Use small sample user-item matrix")
        df = None
        if sample_btn:
            users = [f"user_{i}" for i in range(1, 21)]
            items = [f"item_{j}" for j in range(1, 31)]
            rng = np.random.RandomState(42)
            mat = rng.poisson(1, (len(users), len(items)))
            df = pd.DataFrame(mat, index=users, columns=items)
            st.success("Sample matrix created (20x30).")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded, index_col=0)
                st.success("File loaded (first column used as index).")
                log_consent_action(
                    "upload",
                    {
                        "filename": uploaded.name,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    },
                )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df = None
        if df is not None:
            st.subheader("Preview (top-left)")
            st.dataframe(df.iloc[:8, :8])
            mode = st.radio("Mode", ["user-item", "item-features"], index=0)
            if mode == "user-item":
                user_id = st.text_input("Enter user id (index)", value=str(df.index[0]))
                top_k = st.slider("Top K", 1, 20, 10)
                if st.button("Recommend for user"):
                    try:
                        rec = rec_recommend_items_from_user_item_matrix(
                            df, user_id, top_k=top_k
                        )
                        st.table(rec)
                    except Exception as e:
                        st.error(str(e))
            else:
                item_id = st.text_input(
                    "Enter item id (index)", value=str(df.columns[0])
                )
                top_k = st.slider("Top K", 1, 20, 10)
                if st.button("Find similar items"):
                    try:
                        rec = rec_recommend_similar_items_from_item_features(
                            df, item_id, top_k=top_k
                        )
                        st.table(rec)
                    except Exception as e:
                        st.error(str(e))

# Transport
elif page == "Transport":
    st.header("Transport — Route Optimization")
    consent_widget()
    if not st.session_state["consent_given"]:
        st.info("Give consent to upload and process transport data.")
    else:
        uploaded = st.file_uploader("Upload edges CSV", type=["csv"], key="trans_file")
        sample_btn = st.button("Use small sample graph")
        edges = None
        if sample_btn:
            edges = pd.DataFrame(
                [
                    {"source": "A", "target": "B", "weight": 5},
                    {"source": "B", "target": "C", "weight": 3},
                    {"source": "A", "target": "C", "weight": 10},
                    {"source": "C", "target": "D", "weight": 1},
                ]
            )
            st.success("Sample edges created.")
        if uploaded is not None:
            try:
                edges = pd.read_csv(uploaded)
                st.success("File loaded.")
                log_consent_action(
                    "upload",
                    {
                        "filename": uploaded.name,
                        "rows": int(edges.shape[0]),
                        "cols": int(edges.shape[1]),
                    },
                )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                edges = None
        if edges is not None:
            st.subheader("Preview")
            st.dataframe(edges.head())
            origin = st.text_input("Origin node", value=str(edges.iloc[0]["source"]))
            destination = st.text_input(
                "Destination node", value=str(edges.iloc[-1]["target"])
            )
            if st.button("Compute shortest route"):
                try:
                    G = tr_build_graph_from_edges(edges)
                    out = tr_shortest_route(G, origin, destination)
                    st.json(out)
                    st.write("Path -> " + " -> ".join(out["path"]))
                    st.write(f"Total weight: {out['length']}")
                except Exception as e:
                    st.error(str(e))

# Manufacturing
elif page == "Manufacturing":
    st.header("Manufacturing — Predictive Maintenance")
    consent_widget()
    if not st.session_state["consent_given"]:
        st.info("Give consent to upload and process manufacturing data.")
    else:
        uploaded = st.file_uploader(
            "Upload features CSV", type=["csv"], key="manu_file"
        )
        sample_btn = st.button("Use small sample features")
        df = None
        if sample_btn:
            rng = np.random.RandomState(42)
            n = 300
            df = pd.DataFrame(
                {
                    "mean_all": rng.normal(10, 2, n),
                    "std_all": rng.normal(1, 0.2, n),
                    "max_all": rng.normal(15, 3, n),
                    "min_all": rng.normal(5, 1, n),
                }
            )
            df["target"] = (
                0.1 * (df["max_all"] - df["min_all"]) + rng.normal(0, 1, n) > 1.5
            ).astype(int)
            st.success("Sample manufacturing dataset created.")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success("File loaded.")
                log_consent_action(
                    "upload",
                    {
                        "filename": uploaded.name,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    },
                )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df = None
        if df is not None:
            st.subheader("Preview")
            st.dataframe(df.head())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = st.text_input("Target column name", value="target")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train PM Model"):
                    try:
                        if target_col not in df.columns:
                            st.error(
                                f"Target column '{target_col}' not found. Available columns: {', '.join(df.columns)}"
                            )
                        else:
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            with st.spinner("Training RandomForest..."):
                                res = manu_train_and_save(X, y)
                            st.success(f"Model trained and saved. {res}")
                    except Exception as e:
                        st.error(str(e))
            with col2:
                if st.button("Predict PM Risk"):
                    try:
                        X = df.drop(columns=[target_col])
                        probs = manu_predict_from_features(X)
                        st.subheader("Predicted risk (first 20 rows)")
                        st.table(pd.DataFrame({"risk_score": probs}).head(20))
                    except Exception as e:
                        st.error(str(e))

# Data Inspector (upload + auto-detect)
elif page == "Data Inspector":
    st.header("Data Inspector — upload dataset, auto-detect & summary")
    consent_widget()
    if not st.session_state["consent_given"]:
        st.info("Give consent to upload data and run inspection.")
    else:
        uploaded = st.file_uploader(
            "Upload dataset (CSV / XLSX / PDF)",
            type=["csv", "xls", "xlsx", "pdf"],
            key="inspector_file",
        )
        if uploaded is not None:
            try:
                df = read_uploaded_file_bytes(uploaded)
                st.success("File loaded.")
                log_consent_action(
                    "upload",
                    {
                        "filename": uploaded.name,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    },
                )
                pii = detect_pii_in_df(df)
                if pii:
                    st.warning(
                        "Potential PII detected. Review before saving or sharing."
                    )
                    st.json(pii)
                    if st.button("Mask detected PII (basic)"):
                        for col, info in pii.items():
                            df[col] = (
                                df[col]
                                .astype(str)
                                .apply(
                                    lambda v: (
                                        re.sub(
                                            PII_PATTERNS["email"], "[EMAIL_MASKED]", v
                                        )
                                        if isinstance(v, str)
                                        else v
                                    )
                                )
                            )
                            df[col] = (
                                df[col]
                                .astype(str)
                                .apply(
                                    lambda v: (
                                        re.sub(
                                            PII_PATTERNS["aadhar_like"],
                                            "[PII_MASKED]",
                                            v,
                                        )
                                        if isinstance(v, str)
                                        else v
                                    )
                                )
                            )
                        st.success("Basic masking applied.")
                roles = detect_field_roles(df)
                det = quick_dataset_summary(df, roles=roles)
                st.subheader("Detected columns & roles")
                st.dataframe(
                    pd.DataFrame.from_dict(roles, orient="index")
                    .reset_index()
                    .rename(columns={"index": "column"})
                )
                st.subheader("Quick summary")
                st.info(det["summary_text"])
                st.subheader("Preview")
                st.dataframe(det["preview_df"])
                chosen = None
                if det["candidate_targets"]:
                    chosen = st.selectbox(
                        "Choose target column (auto-suggested)",
                        options=["(none)"] + det["candidate_targets"],
                    )
                else:
                    chosen = st.text_input("Specify target column (if any)", value="")
                st.session_state["inspector_df"] = df
                st.session_state["inspector_roles"] = roles
                st.session_state["inspector_target"] = (
                    chosen if chosen and chosen != "(none)" else None
                )
            except Exception as e:
                st.error("Failed to parse file: " + str(e))

# Resources page
elif page == "Resources":
    st.header("Resources & Examples")
    md_path = os.path.join(ROOT, "resources_bigdata.md")
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf8") as f:
            content = f.read()
        st.markdown(content, unsafe_allow_html=True)
    else:
        st.info(
            "resources_bigdata.md not found. Create it in repo root to show curated big-data examples."
        )

st.markdown("---")
st.markdown("Made for demo: VizClean — single-file multipage app")
