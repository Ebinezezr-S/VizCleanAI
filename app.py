# app.py — VizClean Streamlit app with Hugging Face integration
import os
from pathlib import Path
import io
import traceback

import pandas as pd
import streamlit as st

# Try to import HF Inference client; if not available show helpful message
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# --- Ensure Streamlit can write config inside container/Space ---
os.environ.setdefault("HOME", str(Path.cwd()))
(Path.cwd() / ".streamlit").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

# --- App constants ---
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="VizCleanAI", layout="wide")

st.title("VizClean — Upload, Clean, and (optionally) generate images")

# --- Initialize HF client from HF_TOKEN env var (Space secret) ---
hf_token = os.environ.get("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")
hf_client = None
if HF_AVAILABLE and hf_token:
    try:
        hf_client = InferenceClient(token=hf_token)
    except Exception as e:
        hf_client = None
        st.warning("Could not initialize Hugging Face InferenceClient: " + str(e))

# show token / client status (non-sensitive)
if not HF_AVAILABLE:
    st.info("Hugging Face SDK not installed in the environment. Install `huggingface-hub` to enable model calls.")
elif not hf_token:
    st.info("No HF_TOKEN found in environment. Add it to Space Secrets (name: HF_TOKEN) to enable Inference calls.")
else:
    st.success("Hugging Face client is configured." if hf_client else "Hugging Face client failed to init — check token & package versions.")

# --- File upload & preview ---
st.header("Upload a CSV or Excel file")
uploaded = st.file_uploader("Drag & drop file here (CSV, XLS, XLSX). Limit ~200MB", type=["csv", "xls", "xlsx"])

def read_file_to_df(uploaded_file):
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

if uploaded:
    st.info(f"File: **{uploaded.name}** — {round(uploaded.size/1024, 1)} KB")
    try:
        df = read_file_to_df(uploaded)
        st.write("File read successfully.")
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df.head(5))
        # save a local copy (timestamp-based)
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        local_in_name = OUT_DIR / f"{Path(uploaded.name).stem}_{ts}{Path(uploaded.name).suffix}"
        uploaded.seek(0)
        # stream copy
        with open(local_in_name, "wb") as fh:
            fh.write(uploaded.getbuffer())
        st.write(f"Saved a local copy to `{local_in_name}` (not tracked by git).")
        st.session_state["uploaded_df"] = df
        st.session_state["uploaded_name"] = uploaded.name
    except Exception as e:
        st.error("Failed to read file: " + str(e))
        st.exception(traceback.format_exc())

# --- Basic cleaning UI ---
st.header("Cleaning options")
if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
    drop_empty = st.checkbox("Drop rows where *all* values are NaN", value=True)
    dedupe = st.checkbox("Drop exact duplicate rows", value=True)
    normalize_cols = st.checkbox("Strip column whitespace from column names", value=True)
    run_clean = st.button("Run cleaning steps")
    if run_clean:
        cleaned = df.copy()
        if normalize_cols:
            cleaned.columns = [str(c).strip() for c in cleaned.columns]
        if drop_empty:
            cleaned = cleaned.dropna(how="all")
        if dedupe:
            cleaned = cleaned.drop_duplicates()
        # Save cleaned file
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        orig_name = Path(st.session_state.get("uploaded_name", "uploaded")).stem
        cleaned_name = OUT_DIR / f"{orig_name}_{ts}_cleaned.csv"
        cleaned.to_csv(cleaned_name, index=False)
        st.success("Cleaning finished. Cleaned file saved.")
        st.write(f"Saved cleaned file to `{cleaned_name}`.")
        st.subheader("Cleaning result (sample)")
        st.dataframe(cleaned.head(10))
        st.session_state["cleaned_df"] = cleaned
        st.session_state["cleaned_path"] = str(cleaned_name)

# allow download of cleaned file
if st.session_state.get("cleaned_df") is not None:
    st.header("Download cleaned data")
    cleaned = st.session_state["cleaned_df"]
    cleaned_buf = io.BytesIO()
    cleaned.to_csv(cleaned_buf, index=False)
    cleaned_buf.seek(0)
    st.download_button(label="Download cleaned CSV", data=cleaned_buf, file_name=f"{Path(st.session_state.get('cleaned_path')).name}")

# --- Optional: Hugging Face image generation ---
st.header("Optional: Generate an image from a prompt (Hugging Face)")
with st.expander("Image generator (calls HF Inference)"):
    prompt = st.text_area("Image prompt", "A dragon flying over a medieval castle")
    model_input = st.text_input("Model (card id)", "black-forest-labs/FLUX.1-dev")
    generate = st.button("Generate image")
    if generate:
        if not hf_client:
            st.error("Hugging Face client not configured. Set HF_TOKEN in Space secrets and ensure huggingface-hub is installed.")
        else:
            try:
                st.info("Calling HF Inference... this may take a few seconds.")
                img = hf_client.text_to_image(prompt, model=model_input)
                # Save image robustly
                saved_path = OUT_DIR / "generated_image.png"
                try:
                    # PIL.Image-like object
                    img.save(saved_path)
                except Exception:
                    # maybe bytes or dict with base64
                    if isinstance(img, (bytes, bytearray)):
                        saved_path.write_bytes(img)
                    elif isinstance(img, dict) and "image" in img:
                        import base64
                        data = img["image"]
                        if isinstance(data, (bytes, bytearray)):
                            saved_path.write_bytes(data)
                        else:
                            saved_path.write_bytes(base64.b64decode(data))
                    else:
                        st.error(f"Unknown return type from text_to_image: {type(img)}")
                        saved_path = None
                if saved_path and saved_path.exists():
                    st.image(str(saved_path))
                    st.success(f"Saved image to: {saved_path}")
                else:
                    st.error("Image generation succeeded but file was not saved.")
            except Exception as e:
                st.error("Image generation failed: " + str(e))
                st.exception(traceback.format_exc())

# --- Footer: diagnostics & helpful tips ---
st.markdown("---")
st.markdown(
    "Tips: keep large data files out of the repo (use `data/` which is in `.gitignore`). "
    "Add your HF token in Space Settings → Secrets as `HF_TOKEN`."
)

if st.checkbox("Show debug info"):
    st.subheader("ENV & paths")
    st.write("Working dir:", Path.cwd())
    st.write("OUT_DIR:", OUT_DIR.resolve())
    st.write("HF token present:", bool(hf_token))
    st.write("HF client initialized:", bool(hf_client))
