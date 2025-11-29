import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from matplotlib import cm
from io import BytesIO
from PIL import Image

# -------------------------

# (1) Settings

# -------------------------

st.set_page_config(page_title="🌱 Advanced Land Classification", layout="wide")
image_size = (128, 128)

# -------------------------

# (2) Preprocessing Functions

# -------------------------

def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, image_size)
    img_normalized = img_resized / 255.0
    return img_normalized

def overlay_mask(img, mask, alpha=0.6, threshold=0.5):
    mask_bin = (mask[:,:,0] > threshold)
    overlay = img.copy()
    overlay[mask_bin] = (overlay[mask_bin] * (1-alpha) + np.array([0,255,0]) * alpha).astype(np.uint8)
    return overlay, mask_bin

def heatmap_overlay(img, mask):
    cmap = cm.get_cmap('Greens')
    heatmap = cmap(mask[:,:,0])
    heatmap_img = (heatmap[:,:,:3]*255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.6, heatmap_img, 0.4, 0)
    return overlay

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

st.title("🌱 Advanced Land Classification – Desert vs Agriculture")
uploaded_files = st.file_uploader("Upload one or more aerial images", type=["jpg","png","jpeg"], accept_multiple_files=True)

alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.6)

# -------------------------

# (5) Process Each Image

# -------------------------

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.subheader(f"File: {uploaded_file.name}")
        st.image(img_display, caption="Original Image", use_column_width=True)

        img_pre = preprocess_image(img)
        pred_mask = model.predict(np.expand_dims(img_pre, 0))[0]

        overlay, mask_bin = overlay_mask((img_pre*255).astype(np.uint8), pred_mask, alpha=alpha)
        heatmap = heatmap_overlay((img_pre*255).astype(np.uint8), pred_mask)
        prop_agri = mask_bin.mean()

        st.progress(int(prop_agri*100))
        if prop_agri > 0.05:
            st.success(f"🌱 Suitable for agriculture — Green area: {prop_agri*100:.2f}%")
        else:
            st.warning(f"🏜️ Mostly desert — Green area: {prop_agri*100:.2f}%")

        tab1, tab2, tab3, tab4 = st.tabs(["Overlay Image", "Predicted Mask", "Heatmap", "Original"])
        with tab1:
            st.image(overlay, caption="Image with Mask Overlay", use_column_width=True)
        with tab2:
            st.image(pred_mask[:,:,0], caption="Predicted Mask", clamp=True, width=350)
        with tab3:
            st.image(heatmap, caption="Heatmap Overlay", use_column_width=True)
        with tab4:
            st.image(img_display, caption="Original Image", use_column_width=True)

        # -------------------------
        # Download Buttons
        # -------------------------
        def convert_to_bytes(img_array):
            im_pil = Image.fromarray(img_array)
            buf = BytesIO()
            im_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            return byte_im

        st.download_button("Download Overlay", data=convert_to_bytes(overlay), file_name=f"overlay_{uploaded_file.name}", mime="image/png")
        st.download_button("Download Mask", data=convert_to_bytes((pred_mask[:,:,0]*255).astype(np.uint8)), file_name=f"mask_{uploaded_file.name}", mime="image/png")
