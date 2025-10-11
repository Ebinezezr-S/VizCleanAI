# ===========================================================
# vizclean EJS — Full Streamlit App (with Login/Signup)
# ===========================================================
import hashlib
import json
import os
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ===========================================================
# Configuration
# ===========================================================
st.set_page_config(page_title="vizclean EJS — Controls", layout="wide")

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
for d in [DATA_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ===========================================================
# --- Local Auth Module ---
# ===========================================================
USERS_PATH = DATA_DIR / "users.json"


def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def load_users() -> dict:
    if not USERS_PATH.exists():
        return {}
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf8"))
    except Exception:
        return {}


def save_users(users: dict):
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf8")


def verify_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    rec = users[username]
    return rec.get("pw_hash") == _hash_pw(password, rec.get("salt", ""))


def create_user(username: str, password: str) -> bool:
    if (
        len(password) < 6
        or not re.search(r"[A-Za-z]", password)
        or not re.search(r"\d", password)
    ):
        return False
    users = load_users()
    if username in users:
        return False
    salt = secrets.token_hex(8)
    users[username] = {"salt": salt, "pw_hash": _hash_pw(password, salt)}
    save_users(users)
    return True


def change_password(username: str, old_pw: str, new_pw: str) -> Tuple[bool, str]:
    users = load_users()
    if username not in users:
        return False, "User not found"
    if not verify_user(username, old_pw):
        return False, "Old password incorrect"
    salt = secrets.token_hex(8)
    users[username] = {"salt": salt, "pw_hash": _hash_pw(new_pw, salt)}
    save_users(users)
    return True, "Password changed"


if "user" not in st.session_state:
    st.session_state.user = None
if "auth_msg" not in st.session_state:
    st.session_state.auth_msg = ""


def auth_sidebar():
    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/9075/9075521.png", width=60
    )
    st.sidebar.markdown("## 🔐 Login / Signup")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as **{st.session_state.user}**")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()
        if st.sidebar.button("Change Password"):
            st.session_state.change_pw = True
    else:
        mode = st.sidebar.radio("Action", ["Sign In", "Sign Up"])
        if mode == "Sign In":
            u = st.sidebar.text_input("Username")
            p = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Sign In"):
                if verify_user(u.strip(), p):
                    st.session_state.user = u.strip()
                    st.sidebar.success("Welcome " + u)
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Invalid credentials.")
        else:
            u = st.sidebar.text_input("Choose Username")
            p1 = st.sidebar.text_input("Choose Password", type="password")
            p2 = st.sidebar.text_input("Confirm Password", type="password")
            if st.sidebar.button("Create Account"):
                if p1 != p2:
                    st.sidebar.error("Passwords don’t match.")
                elif create_user(u.strip(), p1):
                    st.sidebar.success("Account created! Please sign in.")
                else:
                    st.sidebar.error("Invalid password or username exists.")
    if "change_pw" in st.session_state and st.session_state.change_pw:
        old = st.sidebar.text_input("Old Password", type="password")
        new = st.sidebar.text_input("New Password", type="password")
        new2 = st.sidebar.text_input("Confirm New Password", type="password")
        if st.sidebar.button("Apply Change"):
            if new != new2:
                st.sidebar.error("Mismatch.")
            else:
                ok, msg = change_password(st.session_state.user, old, new)
                st.sidebar.write(msg)
        if st.sidebar.button("Cancel"):
            st.session_state.change_pw = False


auth_sidebar()
if not st.session_state.user:
    st.warning("Please log in to continue.")
    st.stop()

# ===========================================================
# --- Main App Pages ---
# ===========================================================
st.title("vizclean EJS — Controls")

page = st.sidebar.selectbox(
    "Choose Domain", ["Overview", "Healthcare", "Data Inspector"]
)

# -----------------------------------------------------------
# Overview
# -----------------------------------------------------------
if page == "Overview":
    st.header("Overview")
    st.markdown(
        """
    ### 🌐 Welcome to vizclean EJS
    This demo integrates:
    - **Healthcare Prediction**
    - **Dataset Inspector & Cleaner**
    - Local User Authentication  
    ---
    """
    )
    st.info("Use sidebar to switch between domains.")

# -----------------------------------------------------------
# Healthcare
# -----------------------------------------------------------
elif page == "Healthcare":
    st.header("Healthcare — Disease Prediction Demo")
    st.write("Upload CSV with numeric features + a 'target' column (0/1).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        lower_map = {c.lower(): c for c in df.columns}
        if "outcome" in lower_map and "target" not in df.columns:
            df = df.rename(columns={lower_map["outcome"]: "target"})
            st.info("Auto-renamed 'Outcome' → 'target'")
        st.dataframe(df.head())
        target_col = (
            "target" if "target" in df.columns else st.text_input("Target Column")
        )
        if st.button("Train Model"):
            try:
                X = df.drop(columns=[target_col])
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                st.success(f"✅ Trained Accuracy = {acc:.3f}, ROC AUC = {auc:.3f}")
            except Exception as e:
                st.error(f"Training error: {e}")

# -----------------------------------------------------------
# Data Inspector
# -----------------------------------------------------------
elif page == "Data Inspector":
    st.header("Data Inspector — Upload and Analyze")
    up = st.file_uploader("Upload Dataset (CSV/XLSX)", type=["csv", "xlsx"])
    if up:
        if up.name.endswith(".csv"):
            df = pd.read_csv(up)
        else:
            df = pd.read_excel(up)
        st.write(f"**Rows:** {len(df)}   **Cols:** {len(df.columns)}")
        st.dataframe(df.head())
        st.markdown("### Basic Statistics")
        st.write(df.describe())
        if st.button("Save Copy"):
            out = DATA_DIR / f"inspected_{int(time.time())}.csv"
            df.to_csv(out, index=False)
            st.success(f"Saved to {out}")
            st.download_button(
                "Download Cleaned CSV",
                data=df.to_csv(index=False).encode(),
                file_name="cleaned.csv",
            )

st.markdown("---")
st.caption("© 2025 vizclean EJS — Demo App | For local use only")
