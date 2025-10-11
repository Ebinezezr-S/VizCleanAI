# streamlit_auth_example.py
import streamlit as st

from app.auth import (change_password, create_user, ensure_db, get_user,
                      verify_user)

ensure_db()
st.set_page_config(page_title="VizClean Auth Example")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

if not st.session_state.logged_in:
    st.title("VizClean — Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            ok = verify_user(username, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login successful")
                user = get_user(username)
                if user and user.force_reset:
                    st.warning(
                        "You must change your password (administrator set temporary password)."
                    )
            else:
                st.error("Invalid credentials")
    with col2:
        if st.button("Create demo user"):
            try:
                create_user("demo", "demo1234", role="user", force_reset=False)
                st.success("demo user created: demo/demo1234")
            except Exception as e:
                st.error(str(e))
else:
    st.title(f"Welcome, {st.session_state.user}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.experimental_rerun()

    st.subheader("Change password")
    new1 = st.text_input("New password", type="password")
    new2 = st.text_input("Confirm new password", type="password")
    if st.button("Change password"):
        if new1 and new1 == new2:
            change_password(st.session_state.user, new1)
            st.success("Password changed")
        else:
            st.error("Passwords do not match")
