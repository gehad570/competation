import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from matplotlib import cm
from io import BytesIO
from PIL import Image
import zipfile
import pandas as pd # Moved import to top for better practice

# -------------------------

# (1) Settings

# -------------------------

st.set_page_config(page_title="🌱 Land Classification", layout="wide")
image_size = (128, 128)

# -------------------------

# (2) Preprocessing Functions

# -------------------------

def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, image_size)
    img_normalized = img_resized / 255.0
    return img_normalized

def overlay_mask(img, mask, threshold=0.5):
    mask_bin = (mask[:,:,0] > threshold)
    overlay = img.copy()
    overlay[mask_bin] = (overlay[mask_bin] * 0.4 + np.array([0,255,0]) * 0.6).astype(np.uint8)
    return overlay, mask_bin

def heatmap_overlay(img, mask):
    cmap = cm.get_cmap('Greens')
    heatmap = cmap(mask[:,:,0])
    heatmap_img = (heatmap[:,:,:3]*255).astype(np.uint8)
    # The img passed here is (img_pre*255).astype(np.uint8) which is already RGB
    overlay = cv2.addWeighted(img, 0.6, heatmap_img, 0.4, 0)
    return overlay

def convert_to_bytes(img_array):
    im_pil = Image.fromarray(img_array)
    buf = BytesIO()
    im_pil.save(buf, format="PNG")
    return buf.getvalue()

# -------------------------

# (3) Load Model

# -------------------------

@st.cache_resource
def load_model(model_path="model_unet.h5"):
    return tf.keras.models.load_model(model_path)

try:
    model = load_model("model_unet.h5")
    st.sidebar.success("✅ Model loaded successfully!")
except:
    st.sidebar.error("❌ Could not load model_unet.h5. Upload it to the same folder.")
    st.stop()

# -------------------------

# (4) UI

# -------------------------

st.title("🌱 Professional Land Classification – Desert vs Agriculture")
uploaded_files = st.file_uploader("Upload one or more aerial images", type=["jpg","png","jpeg"], accept_multiple_files=True)

threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5)

# -------------------------

# (5) Process Each Image

# -------------------------

results = []
if uploaded_files:
    st.subheader("Results")
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pre = preprocess_image(img)
        pred_mask = model.predict(np.expand_dims(img_pre, 0))[0]

        # Prepare base image for overlay/heatmap (resized and converted back to 0-255 RGB)
        img_pre_display = (img_pre * 255).astype(np.uint8)

        overlay, mask_bin = overlay_mask(img_pre_display.copy(), pred_mask, threshold=threshold)
        heatmap = heatmap_overlay(img_pre_display.copy(), pred_mask)
        prop_agri = mask_bin.mean()

        results.append({
            "name": uploaded_file.name,
            "overlay": overlay,
            "mask": (pred_mask[:,:,0]*255).astype(np.uint8),
            "heatmap": heatmap,
            "green_area": prop_agri,
            "original_resized": img_pre_display # Store resized original for consistent display
        })

    # -------------------------
    # Summary Table
    # -------------------------
    summary = pd.DataFrame([{"File": r["name"], "Green Area (%)": r["green_area"]*100} for r in results])
    st.table(summary)

    # -------------------------
    # Display Images
    # -------------------------
    for r in results:
        st.subheader(f"File: {r['name']}")
        tab1, tab2, tab3, tab4 = st.tabs(["Overlay", "Mask", "Heatmap", "Original Resized"])
        with tab1:
            st.image(r["overlay"], caption="Overlay", use_column_width=True)
        with tab2:
            st.image(r["mask"], caption="Predicted Mask", clamp=True, width=350)
        with tab3:
            st.image(r["heatmap"], caption="Heatmap Overlay", use_column_width=True)
        with tab4:
            st.image(r["original_resized"], caption="Original Resized", use_column_width=True)

    # -------------------------
    # Download as ZIP
    # -------------------------
    if st.button("Download All Results as ZIP"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for r_download in results:
                zip_file.writestr(f"overlay_{r_download['name']}", convert_to_bytes(r_download["overlay"]))
                zip_file.writestr(f"mask_{r_download['name']}", convert_to_bytes(r_download["mask"]))
                zip_file.writestr(f"heatmap_{r_download['name']}", convert_to_bytes(r_download["heatmap"]))
        st.download_button("Download ZIP", data=zip_buffer.getvalue(), file_name="all_results.zip", mime="application/zip")
