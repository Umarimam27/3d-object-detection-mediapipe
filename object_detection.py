# objectron_app.py — Clean Minimal UI (Only MediaPipe 3D Objectron)
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="3D Object Detection | MediaPipe Objectron",
                   page_icon="🎥",
                   layout="centered")

st.markdown("""
<h1 style="text-align:center;">
🎥 3D Object Detection using MediaPipe Objectron
</h1>
<p style="text-align:center; font-size:18px;">
Detect Cup / Shoe / Chair / Camera in Images
</p>
""", unsafe_allow_html=True)

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Dropdown — Select Model
# -----------------------------
st.subheader("Select Model")
MODEL = st.selectbox("Choose an object type:",
                     ["Cup", "Shoe", "Chair", "Camera"])

# -----------------------------
# Upload Image
# -----------------------------
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Drag & drop or browse files",
                                 type=["jpg", "jpeg", "png", "webp"])

# -----------------------------
# Detection Function
# -----------------------------
def run_objectron(image_rgb):
    objectron = mp_objectron.Objectron(
        static_image_mode=True,
        max_num_objects=5,
        min_detection_confidence=0.3,
        model_name=MODEL
    )
    results = objectron.process(image_rgb)
    objectron.close()
    return results

# -----------------------------
# Run detection
# -----------------------------
if uploaded_file:

    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not load image.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Show input image
        st.markdown("### Input Image")
        st.image(img_rgb, use_container_width=True)

        # Run Objectron
        results = run_objectron(img_rgb)

        # Annotate
        annotated = img_rgb.copy()
        count = 0

        if results.detected_objects:
            count = len(results.detected_objects)
            for obj in results.detected_objects:
                mp_drawing.draw_landmarks(
                    annotated, obj.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(
                    annotated, obj.rotation, obj.translation)

        # Show output
        st.markdown("### Detected 3D Output")
        st.image(annotated, use_container_width=True)

        # Show count
        st.info(f"**Detected {count} {MODEL}(s)** in this image.")
