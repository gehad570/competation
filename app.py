import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import zipfile
import pandas as pd

# -------------------------

# إعداد الصفحة

# -------------------------

st.set_page_config(page_title="🌱 كشف المساحة الخضراء", layout="wide")

# -------------------------

# دوال مساعدة

# -------------------------

def convert_to_bytes(img_array):
im_pil = Image.fromarray(img_array)
buf = BytesIO()
im_pil.save(buf, format="PNG")
return buf.getvalue()

def green_area_from_image(img):
"""كشف البيكسلات الخضراء مع شروط أقل صرامة لزيادة نسبة المساحة الخضراء"""
r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
# شروط أقل صرامة للكشف عن الأخضر
green_mask = (g > 100) & (g > r + 10) & (g > b + 10)
fraction = green_mask.mean()
return green_mask, fraction

def overlay_green(img, green_mask):
overlay = img.copy()
overlay[green_mask] = (overlay[green_mask] * 0.4 + np.array([0,255,0]) * 0.6).astype(np.uint8)
return overlay

# -------------------------

# واجهة المستخدم

# -------------------------

st.title("🌱 كشف المساحة الخضراء في الصور")
uploaded_files = st.file_uploader("ارفع صورة أو أكثر", type=["jpg","png","jpeg"], accept_multiple_files=True)

results = []
if uploaded_files:
st.subheader("النتائج")
for uploaded_file in uploaded_files:
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

```
    green_mask, prop_green = green_area_from_image(img_display)
    overlay = overlay_green(img_display, green_mask)

    results.append({
        "name": uploaded_file.name,
        "overlay": overlay,
        "green_area": prop_green
    })

# عرض جدول النتائج
summary = pd.DataFrame([{"الملف": r["name"], "نسبة المساحة الخضراء (%)": r["green_area"]*100} for r in results])
st.table(summary)

# عرض الصور
for r in results:
    st.subheader(f"الملف: {r['name']}")
    tab1, tab2 = st.tabs(["الصورة مع التراكب", "الصورة الأصلية"])
    with tab1:
        st.image(r["overlay"], caption="الصورة مع التراكب", use_column_width=True)
    with tab2:
        st.image(convert_to_bytes(img_display), caption="الصورة الأصلية", use_column_width=True)

# تحميل النتائج كملف ZIP
if st.button("تحميل كل الصور مع التراكب كملف ZIP"):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        for r in results:
            zip_file.writestr(f"overlay_{r['name']}", convert_to_bytes(r["overlay"]))
    st.download_button("تحميل ZIP", data=zip_buffer.getvalue(), file_name="all_overlays.zip", mime="application/zip")
```
