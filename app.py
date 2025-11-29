import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import zipfile
import pandas as pd

# -------------------------

# Settings

# -------------------------

st.set_page_config(page_title="🌱 Green Area Detection", layout="wide")

# -------------------------

# Preprocessing Functions

# -------------------------

def convert_to_bytes(img_array):
    im_pil = Image.fromarray(img_array)
    buf = BytesIO()
    im_pil.save(buf, format="PNG")
    return buf.getvalue()

def green_area_from_image(img):
    """Detect green pixels with stricter threshold to reduce overestimation"""
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    # stricter green criteria
    green_mask = (g > 120) & (g > r + 20) & (g > b + 20)
    fraction = green_mask.mean()
    return green_mask, fraction

def overlay_green(img, green_mask, alpha=0.6):
    overlay = img.copy()
    overlay[green_mask] = (overlay[green_mask] * (1-alpha) + np.array([0,255,0]) * alpha).astype(np.uint8)
    return overlay

# -------------------------

# UI

# -------------------------

st.title("🌱 Green Area Detection in Images")
uploaded_files = st.file_uploader("Upload one or more images", type=["jpg","png","jpeg"], accept_multiple_files=True)
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.6)

results = []
if uploaded_files:
    st.subheader("Results")
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Green detection
        green_mask, prop_green = green_area_from_image(img_display)
        overlay = overlay_green(img_display, green_mask, alpha=alpha)

        results.append({
            "name": uploaded_file.name,
            "overlay": overlay,
            "green_area": prop_green
        })

    # Summary table
    summary = pd.DataFrame([{"File": r["name"], "Green Area (%)": r["green_area"]*100} for r in results])
    st.table(summary)

    # Display images
    for r in results:
        st.subheader(f"File: {r['name']}")
        tab1, tab2 = st.tabs(["Overlay", "Original"])
        with tab1:
            st.image(r["overlay"], caption="Overlay", use_column_width=True)
        with tab2:
            st.image(convert_to_bytes(img_display), caption="Original", use_column_width=True)

    # Download as ZIP
    if st.button("Download All Overlays as ZIP"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for r in results:
                zip_file.writestr(f"overlay_{r['name']}", convert_to_bytes(r["overlay"]))
        st.download_button("Download ZIP", data=zip_buffer.getvalue(), file_name="all_overlays.zip", mime="application/zip")

