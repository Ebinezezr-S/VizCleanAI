@echo off
cd /d "D:\Document\Data Tool\vizclean_ds_app"
call .\.venv\Scripts\activate

:: Set a token for this session (change value before running)
set API_TOKEN=my-secret-token

:: For current window only (PowerShell uses , but in .bat we export)
setx API_TOKEN "%API_TOKEN%" >nul

:: Start secure server (uses setx value)
uvicorn secure_file_server:app --host 127.0.0.1 --port 8502 --reload
pause
