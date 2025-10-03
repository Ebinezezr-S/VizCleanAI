# --- HF image generator block (paste into main_app.py at the place you want it) ---
import time
import base64
import traceback
from pathlib import Path
import streamlit as st
from huggingface_hub import InferenceClient

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _save_image_obj(img, out_path: Path) -> bool:
    """Try common ways to persist the returned image object."""
    try:
        # PIL.Image-like
        img.save(out_path)
        return True
    except Exception:
        pass
    # bytes-like
    if isinstance(img, (bytes, bytearray)):
        out_path.write_bytes(img)
        return True
    # dict/list patterns
    if isinstance(img, dict):
        for key in ("image", "images", "data"):
            if key in img:
                data = img[key]
                if isinstance(data, (bytes, bytearray)):
                    out_path.write_bytes(data)
                    return True
                if isinstance(data, str):
                    out_path.write_bytes(base64.b64decode(data))
                    return True
    if isinstance(img, list) and len(img) and isinstance(img[0], (bytes, bytearray)):
        out_path.write_bytes(img[0])
        return True
    return False

st.markdown("### AI Image Generator (Hugging Face)")
col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.text_area("Prompt", "A dragon flying over a medieval castle", height=120)
    negative_prompt = st.text_input("Negative prompt (optional)", "")
    model_id = st.text_input("Model id", "black-forest-labs/FLUX.1-dev")
with col2:
    width = st.number_input("Width", min_value=64, max_value=2048, value=512, step=64)
    height = st.number_input("Height", min_value=64, max_value=2048, value=512, step=64)
    steps = st.number_input("Steps", min_value=1, max_value=200, value=30)
    guidance = st.slider("Guidance scale", 0.0, 20.0, 7.5)
    seed = st.number_input("Seed (0 = random)", value=0, step=1)

provider_choice = st.selectbox("Provider (fallback order: auto → hf-inference)", ["auto", "hf-inference"])
generate = st.button("Generate image")

if generate:
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        timestamp = int(time.time())
        filename = f"vizclean_img_{timestamp}.png"
        out_path = DATA_DIR / filename

        st.info(f"Generating image using model `{model_id}` (provider={provider_choice}) — this may take a few seconds.")
        progress_bar = st.progress(0)
        try:
            providers_to_try = [None] if provider_choice == "auto" else [provider_choice]
            # If user chose auto, still include hf-inference as fallback
            if provider_choice == "auto":
                providers_to_try.append("hf-inference")

            saved = False
            last_exc = None
            for i, prov in enumerate(providers_to_try):
                try:
                    # instantiate client for provider (None => auto)
                    client = InferenceClient() if prov is None else InferenceClient(provider=prov)
                    progress_bar.progress(int((i / len(providers_to_try)) * 50) )
                    # call text_to_image with available kwargs
                    img = client.text_to_image(
                        prompt,
                        model=model_id,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        width=int(width),
                        height=int(height),
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        seed=int(seed) if seed != 0 else None,
                    )
                    progress_bar.progress(int((i / len(providers_to_try)) * 75) )
                    # try saving
                    if _save_image_obj(img, out_path):
                        saved = True
                        progress_bar.progress(100)
                        st.success(f"Saved image to `{out_path}` (provider={prov or 'auto'})")
                        break
                    else:
                        last_exc = f"Unknown return type {type(img)}"
                        st.warning(f"Could not save returned object (provider={prov or 'auto'}) — trying next fallback.")
                except Exception as e:
                    last_exc = e
                    st.warning(f"Provider {prov or 'auto'} failed: {e}")
                    traceback.print_exc()
                progress_bar.progress(int(((i+1) / len(providers_to_try)) * 90))
            if not saved:
                st.error("Image generation failed. See logs above.")
                if last_exc:
                    st.text(str(last_exc))
            else:
                # show and offer download
                st.image(str(out_path), use_column_width=True)
                with open(out_path, "rb") as f:
                    st.download_button("Download image", f.read(), file_name=out_path.name, mime="image/png")
        except Exception as e:
            st.error("Unexpected error during generation.")
            st.exception(e)
        finally:
            progress_bar.empty()
# --- end HF image generator block ---
