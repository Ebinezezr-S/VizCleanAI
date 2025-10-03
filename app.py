# ensure Streamlit can write config in container; MUST be before importing streamlit
import os
from pathlib import Path
os.environ.setdefault("HOME", str(Path.cwd()))
Path(".streamlit").mkdir(exist_ok=True, parents=True)
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
