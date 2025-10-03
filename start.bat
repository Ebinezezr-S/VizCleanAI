@echo off
cd /d "%~dp0"
call .\.venv\Scripts\activate

start "VizClean App" streamlit run main_app.py --server.port 8501 --server.address 127.0.0.1
start "VizClean File Server" python file_server.py

echo.
echo =====================================================
echo   🚀 VizClean AI Started!
echo   UI:       http://127.0.0.1:8501
echo   Downloads: http://127.0.0.1:8502/files
echo =====================================================
pause
