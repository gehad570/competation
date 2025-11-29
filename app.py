import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------

# (1) General Settings

# -------------------------

st.set_page_config(page_title="🌱 Land Classification – Desert vs Agriculture", layout="wide")
image_size = (128, 128)  # Model input size

# -------------------------

# (2) Preprocessing Functions

# -------------------------

def preprocess_image(img):
    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image
    img_resized = cv2.resize(img_rgb, image_size)
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    return img_normalized

def overlay_mask(img, mask, threshold=0.5):
    mask_bin = (mask[:,:,0] > threshold)
    overlay = img.copy()
    overlay[mask_bin] = (overlay[mask_bin] * 0.4 + np.array([0,255,0]) * 0.6).astype(np.uint8)
    return overlay, mask_bin

# -------------------------

# (3) Load Model with Caching

# -------------------------

@st.cache_resource
def load_model(model_path="model_unet.h5"):
    return tf.keras.models.load_model(model_path)

# -------------------------

# (4) UI and Threshold

# -------------------------

st.title("🌱 Land Classification – Desert vs Agriculture")
st.write("Upload an aerial image and the model will detect potential agricultural areas.")

threshold = st.sidebar.slider("Mask Threshold", 0.0, 1.0, 0.5)

# -------------------------

# (5) Load Model

# -------------------------

model_path = "model_unet.h5"
try:
    model = load_model(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
except:
    st.sidebar.error("❌ Could not find model_unet.h5 — upload it to the same folder.")
    st.stop()

uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg","png","jpeg"])

# -------------------------

# (6) Prediction and Display

# -------------------------

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_display, caption="Original Image", use_column_width=True)

    img_pre = preprocess_image(img)
    pred_mask = model.predict(np.expand_dims(img_pre, 0))[0]

    overlay, mask_bin = overlay_mask((img_pre*255).astype(np.uint8), pred_mask, threshold=threshold)
    prop_agri = mask_bin.mean()

    st.subheader("Prediction Results")
    st.progress(int(prop_agri*100))
    if prop_agri > 0.05:
        st.success(f"🌱 Suitable for agriculture — Green area: {prop_agri*100:.2f}%")
    else:
        st.warning(f"🏜️ Mostly desert — Green area: {prop_agri*100:.2f}%")

    tab1, tab2 = st.tabs(["Overlay Image", "Predicted Mask"])
    with tab1:
        st.image(overlay, caption="Image with Mask Overlay", use_column_width=True)
    with tab2:
        st.image(pred_mask[:,:,0], caption="Predicted Mask", clamp=True, width=350)
