import streamlit as st
import numpy as np
import cv2
from matplotlib import cm
from io import BytesIO
from PIL import Image
import zipfile
import pandas as pd

# -------------------------

# (1) Settings

# -------------------------

st.set_page_config(page_title="🌱 Green Area Detection", layout="wide")
image_size = (128, 128)  # optional for resizing heatmap/overlay

# -------------------------

# (2) Functions

# -------------------------

def convert_to_bytes(img_array):
    im_pil = Image.fromarray(img_array)
    buf = BytesIO()
    im_pil.save(buf, format="PNG")
    return buf.getvalue()

def green_area_from_image(img):
    """Detect green pixels and return binary mask and fraction"""
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    green_mask = (g > 100) & (g > r) & (g > b)
    fraction = green_mask.mean()
    return green_mask, fraction

def overlay_mask(img, mask, alpha=0.6):
    overlay = img.copy()
    overlay[mask] = (overlay[mask] * (1-alpha) + np.array([0,255,0]) * alpha).astype(np.uint8)
    return overlay

def heatmap_overlay(img, mask):
    cmap = cm.get_cmap('Greens')
    heatmap = cmap(mask.astype(float))
    heatmap_img = (heatmap[:,:,:3]*255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.6, heatmap_img, 0.4, 0)
    return overlay

# -------------------------

# (3) UI

# -------------------------

st.title("🌱 Green Area Detection – Desert vs Agriculture")
uploaded_files = st.file_uploader("Upload one or more images", type=["jpg","png","jpeg"], accept_multiple_files=True)
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.6)

# -------------------------

# (4) Process Each Image

# -------------------------

results = []
if uploaded_files:
    st.subheader("Results")
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect green pixels
        green_mask, green_fraction = green_area_from_image(img_rgb)

        # Create overlays
        overlay = overlay_mask(img_rgb, green_mask, alpha=alpha)
        heatmap = heatmap_overlay(img_rgb, green_mask)

        results.append({
            "name": uploaded_file.name,
            "overlay": overlay,
            "mask": (green_mask*255).astype(np.uint8),
            "heatmap": heatmap,
            "green_area": green_fraction
        })

    # Summary Table
    summary = pd.DataFrame([{"File": r["name"], "Green Area (%)": r["green_area"]*100} for r in results])
    st.table(summary)

    # Display Images
    for r in results:
        st.subheader(f"File: {r['name']}")
        tab1, tab2, tab3, tab4 = st.tabs(["Overlay", "Mask", "Heatmap", "Original"])
        with tab1:
            st.image(r["overlay"], caption="Overlay", use_column_width=True)
        with tab2:
            st.image(r["mask"], caption="Binary Mask", clamp=True, width=350)
        with tab3:
            st.image(r["heatmap"], caption="Heatmap Overlay", use_column_width=True)
        with tab4:
            st.image(img_rgb, caption="Original Image", use_column_width=True)

    # Download as ZI
    if st.button("Download All Results as ZIP"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for r_download in results:
                zip_file.writestr(f"overlay_{r_download['name']}", convert_to_bytes(r_download["overlay"]))
                zip_file.writestr(f"mask_{r_download['name']}", convert_to_bytes(r_download["mask"]))
                zip_file.writestr(f"heatmap_{r_download['name']}", convert_to_bytes(r_download["heatmap"]))
        st.download_button("Download ZIP", data=zip_buffer.getvalue(), file_name="all_results.zip", mime="application/zip")
