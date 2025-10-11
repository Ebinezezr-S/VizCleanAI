# dashboard_launcher.py
"""
Streamlit Dashboard Launcher for vizclean_ds_app
- Lists repo scripts/files
- Allows launching .bat, .ps1 and .py scripts as background processes
- Shows running processes and allows stopping them
- Shows simple file viewer and tail log helper
"""

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

ROOT = Path.cwd()  # should be project root when running streamlit run ...
ALLOWED_RUN_EXT = [".bat", ".ps1", ".py", ".cmd"]

if "processes" not in st.session_state:
    st.session_state.processes = {}  # key: name -> dict{popen, cmd, started_at, path}

st.set_page_config(page_title="VizClean Launcher", layout="wide")

st.title("VizClean — Project Launcher & Manager")
st.markdown(
    """
Use this page to inspect repository files, run scripts (bat/ps1/py), tail logs and stop processes.
**Run Streamlit from the project root** so this file (dashboard_launcher.py) sees the repo:
`cd "D:\\Document\\Data Tool\\vizclean_ds_app" && .\\.venv\\Scripts\\Activate && streamlit run dashboard_launcher.py`
"""
)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Project files")
    st.write(f"Project root: `{ROOT}`")
    # show top-level files and selectable scripts
    files = sorted([p for p in ROOT.iterdir() if p.is_file()])
    scripts = [p for p in files if p.suffix.lower() in ALLOWED_RUN_EXT]
    other_md = [p for p in files if p.suffix.lower() in [".md", ".log", ".txt"]]

    st.subheader("Runnable scripts (.bat, .ps1, .py)")
    if scripts:
        for s in scripts:
            name = s.name
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.write(f"**{name}** — {s.stat().st_size} bytes")
            # run button
            if c2.button("Run", key=f"run_{name}"):
                # determine command
                if name.lower().endswith(".ps1"):
                    cmd = [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(s),
                    ]
                    shell = False
                elif name.lower().endswith(".bat") or name.lower().endswith(".cmd"):
                    # run via cmd /c so it runs in console; start detached is tricky here
                    cmd = ["cmd", "/c", str(s)]
                    shell = False
                elif name.lower().endswith(".py"):
                    # use the same python; user is expected to have python in PATH used by Streamlit
                    cmd = ["python", str(s)]
                    shell = False
                else:
                    cmd = [str(s)]
                    shell = False

                # prevent duplicate run with same name
                if name in st.session_state.processes:
                    st.warning(
                        f"Process for `{name}` already running. Stop it first to restart."
                    )
                else:
                    try:
                        p = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=str(ROOT),
                        )
                        st.session_state.processes[name] = {
                            "popen": p,
                            "cmd": cmd,
                            "started_at": time.time(),
                            "path": str(s),
                        }
                        st.success(f"Launched `{name}` (pid={p.pid})")
                    except Exception as e:
                        st.error(f"Failed to launch `{name}`: {e}")

            # view button
            if c3.button("View", key=f"view_{name}"):
                try:
                    txt = s.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    txt = "(cannot read file)"
                st.code(
                    txt,
                    language=(
                        "powershell"
                        if name.lower().endswith(".ps1")
                        or name.lower().endswith(".bat")
                        else "python"
                    ),
                )

    else:
        st.info("No runnable scripts (.bat, .ps1, .py) found at root.")

    st.markdown("---")
    st.subheader("Other docs (README / md / logs)")
    if other_md:
        choice = st.selectbox(
            "Open file", options=[p.name for p in other_md], key="other_choice"
        )
        if choice:
            fp = ROOT / choice
            try:
                txt = fp.read_text(encoding="utf-8", errors="replace")
                st.markdown(f"### `{choice}`")
                if fp.suffix.lower() == ".md":
                    st.markdown(txt, unsafe_allow_html=True)
                else:
                    st.code(txt)
            except Exception as e:
                st.error(f"Cannot read file: {e}")
    else:
        st.info("No README/MD/TXT/LOG files found at root.")

with col2:
    st.header("Running processes")
    if st.session_state.processes:
        for name, meta in list(st.session_state.processes.items()):
            p: subprocess.Popen = meta["popen"]
            running = p.poll() is None
            pid = getattr(p, "pid", None)
            started = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(meta["started_at"])
            )
            st.write(
                f"**{name}** — pid={pid} — started: {started} — running: {running}"
            )
            c1, c2 = st.columns([2, 1])
            if c1.button("Tail stdout (10 lines)", key=f"tailout_{name}"):
                try:
                    # read what is currently in stdout (non-blocking)
                    out = meta["popen"].stdout
                    if out is None:
                        st.info("No stdout captured.")
                    else:
                        out.flush()
                        out.seek(0)
                        data = (
                            out.read().decode(errors="replace")
                            if hasattr(out, "read")
                            else "(no data)"
                        )
                        lines = data.splitlines()[-10:]
                        st.code("\n".join(lines))
                except Exception as e:
                    st.error(f"Cannot read stdout: {e}")
            if c2.button("Stop", key=f"stop_{name}"):
                try:
                    if p.poll() is None:
                        p.terminate()
                        time.sleep(1)
                        if p.poll() is None:
                            # force kill
                            if os.name == "nt":
                                subprocess.run(
                                    ["taskkill", "/F", "/PID", str(p.pid)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                )
                            else:
                                p.kill()
                        st.success(f"Terminated {name}")
                    else:
                        st.info(f"{name} already stopped.")
                except Exception as e:
                    st.error(f"Error stopping {name}: {e}")
                # cleanup
                try:
                    del st.session_state.processes[name]
                except Exception:
                    pass
    else:
        st.info("No processes started from this launcher yet.")

st.markdown("---")
st.header("Utilities")

# quick search for files in the repo
q = st.text_input(
    "Search filename pattern (simple substring)",
    value="",
    help="e.g. 'streamlit' or 'start' or 'models'",
)
if q:
    found = [
        str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if q.lower() in p.name.lower()
    ]
    st.write(f"Found {len(found)} matches")
    for f in sorted(found)[:200]:
        st.write(f)

st.markdown("### Models & data")
models_dir = ROOT / "models"
data_dir = ROOT / "data"
st.write("Models dir:", models_dir if models_dir.exists() else "(missing)")
st.write("Data dir:", data_dir if data_dir.exists() else "(missing)")

if models_dir.exists():
    mdl = [p.name for p in models_dir.iterdir() if p.is_file()]
    st.write("Model files:", mdl if mdl else "(none)")

st.markdown("---")
st.caption(
    "Launcher runs scripts in background attached to this Streamlit process. If you close Streamlit, background processes may remain; use OS tools to find/stop them if needed."
)
