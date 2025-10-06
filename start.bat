@echo off
REM Start VizClean AI (local venv) - double-click this file
SETLOCAL
cd /d "%~dp0"

if exist .venv (
  echo Activating virtual env .venv...
  call .venv\Scripts\activate.bat
) else (
  echo Creating virtualenv .venv...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  echo Installing requirements...
  pip install -r requirements.txt
)

echo Starting Streamlit app...
python -m streamlit run streamlit_app.py --server.port=8501 --server.headless=true

ENDLOCAL
