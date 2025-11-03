import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import time
import os

st.set_page_config(page_title="Real-Time Face Expression Recognition", layout="wide")
st.title("Real-Time Face Expression Recognition")

@st.cache_resource
def load_emotion_model():
    model_path = "models/emotion_recognition_model.keras"
    return load_model(model_path)

model = load_emotion_model()

class_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))
    preds = model.predict(reshaped, verbose=0)[0]
    return class_labels[np.argmax(preds)], np.max(preds)

mode = st.radio("Choose Input Mode:", ["Live Camera", "Upload Image"], horizontal=True)

FRAME_WINDOW = st.image(np.zeros((480, 640, 3), dtype=np.uint8))

if mode == "Live Camera":
    col1, col2 = st.columns(2)
    start = col1.button("Start Camera")
    stop = col2.button("Stop Camera")

    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    camera = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            label, confidence = predict_emotion(face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.05)

    camera.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

else:
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_image.read())
        img = cv2.imread(tfile.name)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            label, confidence = predict_emotion(face_img)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        os.remove(tfile.name)

