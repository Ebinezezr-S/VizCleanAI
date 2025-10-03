# ensure Streamlit writes to a writable folder in container/Space
import os
from pathlib import Path

# set HOME to current working dir so Streamlit won't try to write to '/.streamlit'
os.environ.setdefault("HOME", str(Path.cwd()))
# optionally set XDG_CONFIG_HOME instead:
# os.environ.setdefault("XDG_CONFIG_HOME", str(Path.cwd()))

# create .streamlit so Streamlit can write config (and optionally add config.toml)
(Path.cwd() / ".streamlit").mkdir(parents=True, exist_ok=True)

# quiet usage stats
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

# Now import your existing app. If your app's UI code runs at import time (typical),
# this will show your app in the Space. If your app exposes a function (e.g. main()),
# you can call it here instead.
try:
    # prefer app.py (you have an app.py in repo)
    import app  # noqa: E402,F401
except Exception:
    # fallback: try main_app
    try:
        import main_app  # noqa: E402,F401
    except Exception:
        # If neither import works, show a helpful message in Streamlit UI
        import streamlit as st
        st.error("Could not import app.py or main_app.py — please ensure your Streamlit app file is present.")
