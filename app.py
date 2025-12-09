import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------------------------------------
# Load trained model
# -----------------------------------------------------------
model = tf.keras.models.load_model("final_leaf_classifier.keras", compile=False)

# -----------------------------------------------------------
# CLASS NAMES (Paste yours here)
# -----------------------------------------------------------
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
'Tomato___healthy']


# -----------------------------------------------------------
# Helper: Extract Species, Disease & Health
# -----------------------------------------------------------
def parse_label(label):
    parts = label.split("___")
    species = parts[0]

    # Determine Healthy or Diseased
    if "healthy" in parts[1].lower():
        disease = "No Disease"
        health = "Healthy"
    else:
        disease = parts[1]
        health = "Diseased"

    return species, disease, health


# -----------------------------------------------------------
# Prediction Function
# -----------------------------------------------------------
def predict_leaf(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)

    predictions = model.predict(img_arr)[0]
    idx = np.argmax(predictions)

    predicted_label = class_names[idx]
    species, disease, health = parse_label(predicted_label)
    confidence = predictions[idx] * 100

    return species, disease, health, confidence, img


# -----------------------------------------------------------
# Streamlit App UI
# -----------------------------------------------------------
st.set_page_config(page_title="Leaf Classifier", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #2e8b57;'>🌿 Leaf Species & Disease Classifier</h1>
    <p style='text-align: center; font-size: 18px;'>
        <b>By Haseeb Ali (241-5D-DIP-Project)</b><br>
        Instructor: <b>Sir Syed Karar Haider Bukhari</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.write("Upload a clear leaf image to identify **species**, **disease name**, and its **health condition**.")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])


# -----------------------------------------------------------
# Display + Predict
# -----------------------------------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="📸 Uploaded Leaf Image", width=400)

    if st.button("🔍 Predict"):
        with st.spinner("🔄 Analyzing image... Please wait..."):
            species, disease, health, confidence, processed_img = predict_leaf(uploaded_file)

        st.subheader("🔎 Prediction Results")
        st.image(processed_img, caption="Processed Image", width=300)

        st.success(f"🌱 **Species:** {species}")
        st.info(f"🦠 **Disease:** {disease}")

        if health == 'Diseased':
            st.error(f"💊 **Health Status:** {health}")
        else:
            st.success(f"💊 **Health Status:** {health}")

        st.info(f"📊 **Confidence Score:** {confidence:.2f}%")
        st.progress(float(confidence) / 100)
