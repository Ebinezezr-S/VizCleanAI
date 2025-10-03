# vizclean_ds_app

![CI](https://github.com/Ebinezezr-S/vizclean_ds_app/actions/workflows/ci.yml/badge.svg)

Small app to clean and serve visualization-ready data. This repository contains the app code, demo model/report artifacts and helper scripts. Local generated data files are intentionally kept out of Git and stored in a local `data/` directory.

## Contents
- `app/`, `app.py`, `main_app.py` — application code
- `file_server.py`, `secure_file_server.py` — file serving utils
- `start*.bat`, `stop_secure_server.bat` — convenience scripts for Windows
- `deploy/`, `models/`, `reports/` — demo artifacts
- `data/` (ignored) — local cleaned data files (not tracked)

## Prerequisites
- Windows (PowerShell)
- Python 3.8+ (use a virtual environment)
- `pip` available

## Setup (recommended)
1. Create and activate a venv:
   ```powershell
   cd "D:\Document\Data Tool\vizclean_ds_app"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
