import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -----------------------------------------------------------
# Download Leaf / Non-Leaf Model from Google Drive
# -----------------------------------------------------------
os.makedirs("models", exist_ok=True)

LEAF_MODEL_URL = "https://drive.google.com/uc?id=1vNb7pc1XWo68huYL8I9DYpUuP4yby7sl"
LEAF_MODEL_PATH = "models/leaf_nonleaf_classifier.keras"

if not os.path.exists(LEAF_MODEL_PATH):
    with st.spinner("⬇️ Downloading Leaf / Non-Leaf model..."):
        gdown.download(LEAF_MODEL_URL, LEAF_MODEL_PATH, quiet=False)

# -----------------------------------------------------------
# Load Models
# -----------------------------------------------------------
leaf_detector = tf.keras.models.load_model(
    LEAF_MODEL_PATH,
    compile=False
)

disease_model = tf.keras.models.load_model(
    "final_leaf_classifier.keras",   # keep local or host later if needed
    compile=False
)

# -----------------------------------------------------------
# CLASS NAMES (Disease Model)
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
# Helper: Parse Disease Label
# -----------------------------------------------------------
def parse_label(label):
    plant, status = label.split("___")
    if "healthy" in status.lower():
        return plant, "No Disease", "Healthy"
    return plant, status, "Diseased"

# -----------------------------------------------------------
# Step 1: Leaf / Non-Leaf Detection
# -----------------------------------------------------------
def detect_leaf(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prob = leaf_detector.predict(img_arr, verbose=0)[0][0]

    # Training convention: leaf = 0, non_leaf = 1
    if prob < 0.5:
        return True, (1 - prob) * 100, img
    else:
        return False, prob * 100, img

# -----------------------------------------------------------
# Step 2: Disease Detection (ONLY if Leaf)
# -----------------------------------------------------------
def predict_disease(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = disease_model.predict(img_arr, verbose=0)[0]
    idx = np.argmax(preds)

    label = class_names[idx]
    species, disease, health = parse_label(label)
    confidence = preds[idx] * 100

    return species, disease, health, confidence, img

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="Leaf Disease Detection System", layout="centered")

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
    "Upload an image. The system will first detect **Leaf / Non-Leaf**. "
    "If a leaf is detected, it will then classify **species, disease, and health status**."
)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------------
# Prediction Pipeline
# -----------------------------------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="📸 Uploaded Image", width=400)

    if st.button("🔍 Predict"):
        with st.spinner("🔄 Checking if image is a leaf..."):
            is_leaf, leaf_conf, img = detect_leaf(uploaded_file)

        # ❌ Non-Leaf Case
        if not is_leaf:
            st.error("🚫 This image is NOT a leaf.")
            st.image(img, width=300)
            st.info(f"📊 Non-Leaf Confidence: {leaf_conf:.2f}%")

        # ✅ Leaf Case → Disease Detection
        else:
            st.success("🌿 Leaf detected successfully")
            st.info(f"📊 Leaf Confidence: {leaf_conf:.2f}%")

            with st.spinner("🦠 Detecting disease..."):
                species, disease, health, conf, img2 = predict_disease(uploaded_file)

            st.subheader("🔎 Disease Prediction Results")
            st.image(img2, width=300)

            st.success(f"🌱 **Species:** {species}")
            st.warning(f"🦠 **Disease:** {disease}")

            if health == "Diseased":
                st.error(f"💊 **Health Status:** {health}")
            else:
                st.success(f"💊 **Health Status:** {health}")

            st.info(f"📊 **Disease Confidence:** {conf:.2f}%")
            progress_value = float(conf) / 100.0
            progress_value = max(0.0, min(progress_value, 1.0))
            st.progress(progress_value)
            st.subheader("🤖 AI Disease Explanation")

            with st.spinner("🔍 AI is generating disease details..."):
            ai_info = get_disease_info(species, disease)

            st.markdown(ai_info)



def get_disease_info(species, disease):
    prompt = f"""
    Explain the plant disease '{disease}' found in {species}.
    Provide:
    1. Cause
    2. Symptoms
    3. Prevention
    4. Treatment
    Use simple language for farmers/students.
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
        json=payload
    )

    return response.json()["choices"][0]["message"]["content"]


