# \# VizClean - Developer Quickstart

# 

# \## Overview

# VizClean is a local Streamlit app that cleans CSV data and exposes cleaned outputs via a small FastAPI file server.  

# \- UI: Streamlit app — `http://127.0.0.1:8501`  

# \- Files API: FastAPI file server — `http://127.0.0.1:8502/files`

# 

# \## Files in repo

# \- `main\_app.py` — Streamlit application (upload, clean, visuals, download buttons)  

# \- `file\_server.py` — unsecured FastAPI file server serving `deploy/`  

# \- `secure\_file\_server.py` — token-protected FastAPI file server (see below)  

# \- `start.bat` / `start\_all.bat` — convenience launcher(s)  

# \- `start\_file\_server.bat` — file-server-only launcher  

# \- `deploy/` — files served for download  

# \- `deploy/archive/` — archived cleaned outputs (recommended to `.gitignore`)  

# \- `.venv/` — virtualenv (do not commit)

# 

# \## Setup (one-time)

# 1\. Open PowerShell (recommended: \*\*Run as Administrator\*\*)  

# 2\. Go to project root:

# ```powershell

# cd "D:\\Document\\Data Tool\\vizclean\_ds\_app"



