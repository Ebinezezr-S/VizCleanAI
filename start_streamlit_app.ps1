# start_streamlit_app.ps1
cd "D:\Document\Data Tool\vizclean_ds_app"
if (-Not (Test-Path .venv)) {
    python -m venv .venv
}
.\.venv\Scripts\Activate
pip install -r requirements.txt
streamlit run ".\streamlit_modules_app.py"
