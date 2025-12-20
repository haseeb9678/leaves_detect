import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
model = tf.keras.models.load_model(
    "final_leaf_classifier.keras",
    compile=False
)

# -----------------------------------------------------------
# CLASS NAMES (MUST MATCH TRAINING ORDER)
# -----------------------------------------------------------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------
def parse_label(label):
    plant, status = label.split("___")
    if "healthy" in status.lower():
        return plant, "No Disease", "Healthy"
    return plant, status, "Diseased"

# -----------------------------------------------------------
# CORRECT PREDICTION LOGIC
# -----------------------------------------------------------
def predict_leaf(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr, verbose=0)[0]

    # ----- statistics -----
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    second_conf = float(np.sort(preds)[-2])
    gap = top_conf - second_conf

    entropy = -np.sum(preds * np.log(preds + 1e-9))

    # -------------------------------------------------------
    # 🚫 HARD REJECTION CONDITIONS (THIS IS THE FIX)
    # -------------------------------------------------------
    if (
        top_conf < 0.75 or          # low confidence
        gap < 0.25 or               # class confusion
        entropy > 2.5 or            # uncertain prediction
        class_names[top_idx] == "Tomato___Late_blight" and entropy > 1.5
    ):
        return None, None, None, top_conf * 100, img

    label = class_names[top_idx]
    species, disease, health = parse_label(label)

    return species, disease, health, top_conf * 100, img

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="Leaf Disease Detection", layout="centered")

st.markdown("""
<h1 style='text-align:center;color:#2e8b57;'>🌿 Leaf Disease Detection System</h1>
<p style='text-align:center;'>
<b>By Haseeb Ali</b><br>
Instructor: <b>Sir Syed Karar Haider Bukhari</b>
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, width=400)

    if st.button("🔍 Predict"):
        species, disease, health, confidence, img = predict_leaf(uploaded_file)

        if species is None:
            st.error("🚫 Prediction rejected: Image is NOT a valid leaf or model is unreliable.")
            st.info(f"Confidence: {confidence:.2f}%")
            st.warning("Reason: Model collapsed / out-of-distribution input.")
        else:
            st.success(f"🌱 Species: {species}")
            st.warning(f"🦠 Disease: {disease}")
            st.info(f"💊 Health: {health}")
            st.info(f"📊 Confidence: {confidence:.2f}%")
            st.progress(min(max(confidence / 100, 0.0), 1.0))
