import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
from io import BytesIO

# -----------------------------------------------------------
# Load Trained Model
# -----------------------------------------------------------
model = tf.keras.models.load_model("final_leaf_classifier.keras", compile=False)

# -----------------------------------------------------------
# Class Names (Paste Yours Here)
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
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def parse_label(label):
    species = label.split("___")[0]
    health = "Healthy" if "healthy" in label.lower() else "Diseased"
    return species, health


def predict_leaf(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)[0]
    idx = np.argmax(preds)

    predicted_label = class_names[idx]
    species, health = parse_label(predicted_label)
    confidence = preds[idx] * 100

    return species, health, confidence, img


def generate_report(species, health, confidence):
    report = (
        f"Leaf Classification Report\n"
        f"----------------------------\n"
        f"Species: {species}\n"
        f"Health Status: {health}\n"
        f"Confidence: {confidence:.2f}%\n"
    )
    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)
    return buffer


# -----------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------
st.markdown("""
<style>
/* Center Title */
h1 {
    text-align: center;
}

/* Card Container */
.card {
    background-color: #1f1f1f0d;
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid #44444420;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}

/* Confidence Bar Animation */
.confidence-bar {
    height: 25px;
    border-radius: 10px;
    animation: animateBar 1.4s ease-out;
}
@keyframes animateBar {
    from { width: 0%; }
    to { width: 100%; }
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------

st.set_page_config(page_title="Leaf Classifier", layout="wide")

st.markdown(
    """
    <h1 style='color: #2e8b57;'>🌿 Leaf Species & Disease Classifier</h1>
    <p style='text-align:center;font-size:18px;'>
        <b>By: Haseeb Ali – DIP Project</b> <br>
        Instructor: <b>Sir Syed Karar Haider Bukhari</b>
    </p>
    """, unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📸 Uploaded Image")
        st.image(uploaded_file, width=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.button("🔍 Analyze Leaf"):
            with st.spinner("Analyzing leaf... Please wait..."):
                species, health, confidence, processed_img = predict_leaf(uploaded_file)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔎 Analysis Results")

            st.write(f"### 🌱 Species: **{species}**")

            if health == "Healthy":
                st.success(f"💚 Health Status: {health}")
            else:
                st.error(f"💔 Health Status: {health}")

            # Confidence Progress Bar (Animated)
            st.write("### 📊 Confidence")
            st.progress(float(confidence) / 100)

            st.markdown("</div>", unsafe_allow_html=True)

            # Download Report
            report_file = generate_report(species, health, confidence)
            st.download_button(
                label="📄 Download Report",
                data=report_file,
                file_name="leaf_report.txt",
                mime="text/plain"
            )
