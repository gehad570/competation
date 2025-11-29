import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -------------------------

# Page config

# -------------------------

st.set_page_config(
page_title="Agriculture Suitability Detector 🌱",
page_icon="🌿",
layout="centered"
)

st.title("🌱 Agriculture Suitability Detector")
st.write("""
Upload an aerial image of a land area, and the app will predict which parts
are suitable for agriculture using a U-Net segmentation model.
Green areas indicate suitable regions.
""")

# -------------------------

# Load the trained model

# -------------------------

@st.cache_resource(show_spinner=True)
def load_unet_model():
model = tf.keras.models.load_model("simple_unet_model.h5", compile=False)
return model

model = load_unet_model()
st.success("✅ Model loaded successfully!")

# -------------------------

# Image preprocessing

# -------------------------

image_size = (128, 128)

def preprocess_image(img):
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, image_size)
img = img / 255.0
return img

# -------------------------

# Show predictions

# -------------------------

def show_prediction(img, pred_mask):
pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
overlay = (img*255).astype(np.uint8).copy()
overlay[pred_mask_bin[:,:,0]==1] = (overlay[pred_mask_bin[:,:,0]==1]*0.4 + np.array([0,255,0])*0.6).astype(np.uint8)

```
fig, ax = plt.subplots(1,3, figsize=(15,5))

ax[0].imshow((img*255).astype(np.uint8))
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(pred_mask[:,:,0], cmap='gray', vmin=0, vmax=1)
ax[1].set_title("Predicted Mask")
ax[1].axis('off')

ax[2].imshow(overlay)
ax[2].set_title("Overlay (Green = Suitable for Agriculture)")
ax[2].axis('off')

st.pyplot(fig)
```

# -------------------------

# File uploader

# -------------------------

uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
img_preprocessed = preprocess_image(img)

```
# Predict mask
pred_mask = model.predict(np.expand_dims(img_preprocessed,0))[0]
pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

# Calculate proportion of suitable area
prop_agri = np.sum(pred_mask_bin) / (pred_mask_bin.shape[0]*pred_mask_bin.shape[1])
if prop_agri > 0.05:
    st.success(f"🌱 This area is suitable for agriculture ({prop_agri*100:.2f}% green detected).")
else:
    st.warning(f"🏜️ This area is mostly desert ({prop_agri*100:.2f}% green detected).")

# Show predictions
show_prediction(img_preprocessed, pred_mask)
```
