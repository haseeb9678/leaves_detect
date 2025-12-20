import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import requests

# ===========================================================
# PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="Leaf Disease Detection System CNN",
    layout="centered"
)

# ===========================================================
# DOWNLOAD LEAF / NON-LEAF MODEL
# ===========================================================
os.makedirs("models", exist_ok=True)

LEAF_MODEL_URL = "https://drive.google.com/uc?id=1vNb7pc1XWo68huYL8I9DYpUuP4yby7sl"
LEAF_MODEL_PATH = "models/leaf_nonleaf_classifier.keras"

if not os.path.exists(LEAF_MODEL_PATH):
    with st.spinner("⬇️ Downloading Leaf / Non-Leaf model..."):
        gdown.download(LEAF_MODEL_URL, LEAF_MODEL_PATH, quiet=False, fuzzy=True)

# ===========================================================
# LOAD MODELS (CACHED – VERY IMPORTANT)
# ===========================================================
@st.cache_resource
def load_models():
    leaf_model = tf.keras.models.load_model(LEAF_MODEL_PATH, compile=False)
    disease_model = tf.keras.models.load_model(
        "final_leaf_classifier.keras",
        compile=False
    )
    return leaf_model, disease_model

leaf_detector, disease_model = load_models()

# ===========================================================
# CLASS NAMES (EXACT TRAINING ORDER)
# ===========================================================
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

def get_disease_info(species, disease):
    prompt = f"""
    Explain the plant disease '{disease}' found in {species}.
    Provide:
    1. Cause
    2. Symptoms
    3. Prevention
    4. Treatment
    Use simple and clear language.
    """

    headers = {
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )

    return response.json()["choices"][0]["message"]["content"]


# ===========================================================
# HELPERS
# ===========================================================
def parse_label(label):
    plant, status = label.split("___")
    if "healthy" in status.lower():
        return plant, "No Disease", "Healthy"
    return plant, status, "Diseased"

# ---------------- LEAF MODEL PREPROCESS (NORMALIZED) ----------------
def preprocess_leaf_model(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0     # ✅ normalized
    arr = np.expand_dims(arr, axis=0)
    return img, arr

# ---------------- DISEASE MODEL PREPROCESS (RAW PIXELS) -------------
def preprocess_disease_model(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)             # ❗ NO normalization
    arr = np.expand_dims(arr, axis=0)
    return img, arr

# ===========================================================
# STEP 1 — LEAF / NON-LEAF
# ===========================================================
def detect_leaf(uploaded_file):
    img, arr = preprocess_leaf_model(uploaded_file)
    prob = float(leaf_detector.predict(arr, verbose=0)[0][0])

    if prob < 0.5:
        return True, (1 - prob) * 100, img
    else:
        return False, prob * 100, img

# ===========================================================
# STEP 2 — DISEASE DETECTION
# ===========================================================
def predict_disease(uploaded_file):
    img, arr = preprocess_disease_model(uploaded_file)

    preds = disease_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))

    label = class_names[idx]
    species, disease, health = parse_label(label)
    confidence = float(preds[idx] * 100)
    # Top-3 predictions
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(class_names[i], float(preds[i] * 100)) for i in top3_idx]


    return species, disease, health, confidence, img, top3

# ===========================================================
# UI
# ===========================================================
st.markdown(
    """
    <h1 style='text-align:center; color:#2e8b57;'>🌿 Leaf Disease Detection System</h1>
    <p style='text-align:center; font-size:16px;'>
        <b>By Haseeb Ali (241-5D-DIP Project)</b><br>
        Instructor: <b>Sir Syed Karar Haider Bukhari</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload an image. The system first checks **Leaf / Non-Leaf**, "
    "then detects **species, disease, and health status**."
)

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ===========================================================
# PIPELINE
# ===========================================================
if uploaded_file is not None:
    st.image(uploaded_file, caption="📸 Uploaded Image", width=400)

    if st.button("🔍 Predict"):
        # ---------- Leaf Detection ----------
        with st.spinner("🌿 Checking if image is a leaf..."):
            is_leaf, leaf_conf, _ = detect_leaf(uploaded_file)

        if not is_leaf:
            st.error("🚫 This image is NOT a leaf.")
            st.info(f"📊 Non-Leaf Confidence: {leaf_conf:.2f}%")

        else:
            st.success("🌿 Leaf detected successfully")
            st.info(f"📊 Leaf Confidence: {leaf_conf:.2f}%")

            # ---------- Disease Detection ----------
            with st.spinner("🦠 Detecting disease..."):
                species, disease, health, conf, img2, top3 = predict_disease(uploaded_file)

            st.subheader("🔎 Disease Prediction Results")
            st.image(img2, width=300)

            st.success(f"🌱 **Species:** {species}")
            st.warning(f"🦠 **Disease:** {disease}")

            if health == "Diseased":
                st.error(f"💊 **Health Status:** {health}")
            else:
                st.success(f"💊 **Health Status:** {health}")

            st.info(f"📊 **Disease Confidence:** {conf:.2f}%")

            progress_value = max(0.0, min(conf / 100.0, 1.0))
            st.progress(progress_value)
            # ---------------- TOP-3 PREDICTIONS ----------------
            st.subheader("🔝 Top-3 Disease Predictions")
            for name, score in top3:
                  st.write(f"• **{name}** → {score:.2f}%")

            # ---------------- AI EXPLANATION ----------------
            st.subheader("🤖 AI Disease Explanation")

            with st.spinner("🔍 Generating disease explanation..."):
                  ai_text = get_disease_info(species, disease)

            st.markdown(ai_text)


