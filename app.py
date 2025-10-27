import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model("models/basic_cnn_best.keras")

# Load class labels
with open("classes.json", "r") as f:
    class_labels = json.load(f)

# Streamlit UI
st.set_page_config(page_title="Facial Expression Recognition", page_icon="😊", layout="centered")

st.title("Facial Expression Recognition")
st.markdown("Upload an image and let the model predict your emotion!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # same size as your model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100

    st.subheader(f"Predicted Expression: **{class_labels[str(pred_class)]}**")
    st.write(f"Confidence: {confidence:.2f}%")