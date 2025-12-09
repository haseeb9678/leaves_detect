import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# -----------------------------------------------------------
# Load trained model
# -----------------------------------------------------------
model = tf.keras.models.load_model("final_leaf_classifier.keras", compile=False)

# -----------------------------------------------------------
# PASTE YOUR CLASS NAMES HERE
# Example:
# class_names = ["Apple___healthy", "Apple___Black_rot", ...]
# -----------------------------------------------------------
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']



# -----------------------------------------------------------
# Helper function to parse species & health
# -----------------------------------------------------------
def parse_label(label):
    species = label.split("___")[0]
    health = "Healthy" if "healthy" in label.lower() else "Diseased"
    return species, health

# -----------------------------------------------------------
# Prediction function
# -----------------------------------------------------------
def predict_leaf(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)

    predictions = model.predict(img_arr)[0]
    idx = np.argmax(predictions)

    predicted_label = class_names[idx]
    species, health = parse_label(predicted_label)
    confidence = predictions[idx] * 100

    return species, health, confidence

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="Leaf Classification App", layout="centered")

st.title("🌿 Leaf Species & Disease Classifier")
st.write("By Haseeb Ali (241-5D-DIP-Project) **Instructor: (Sir. Syed Karar Haider Bukhari)**.")
st.write("Upload a leaf image to identify **species** and **health condition**.")


uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------------
# Display and Predict
# -----------------------------------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            species, health, confidence = predict_leaf(uploaded_file)

        st.success(f"🌱 **Species:** {species}")
        st.success(f"💊 **Health Status:** {health}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")
