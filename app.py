import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import json
import gdown
import os
from gtts import gTTS
import tempfile
from PIL import Image

st.set_page_config(page_title="BrailleVision", page_icon="⠃", layout="centered")

@st.cache_resource
def load_resources():
    if not os.path.exists("braille_cnn.onnx"):
        gdown.download("https://drive.google.com/uc?id=1gV_PIj8z7Qxu9TdAExfQyqFQw2e2mRAM", "braille_cnn.onnx", quiet=False)
    if not os.path.exists("reverse_map.json"):
        gdown.download("https://drive.google.com/uc?id=15SyxE0eyfBg7uNO7qUog4hhG8d2udfTZ", "reverse_map.json", quiet=False)
    session = ort.InferenceSession("braille_cnn.onnx")
    with open("reverse_map.json") as f:
        reverse_map = json.load(f)
    return session, reverse_map

session, reverse_map = load_resources()

def predict_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (50, 50))
    normalized = resized / 255.0
    ready = normalized.reshape(1, 50, 50, 1).astype(np.float32)
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: ready})[0]
    class_index = str(np.argmax(prediction))
    letter = reverse_map[class_index]
    confidence = float(np.max(prediction)) * 100
    return letter, confidence

def find_and_predict_all(image_array):
    h, w = image_array.shape[:2]

    gray_full = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    _, thresh_full = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_sums = np.sum(thresh_full, axis=1)
    col_sums = np.sum(thresh_full, axis=0)

    dot_rows = np.where(row_sums > 50)[0]
    dot_cols = np.where(col_sums > 50)[0]

    if len(dot_rows) == 0 or len(dot_cols) == 0:
        return "", [], image_array

    # Find column gaps to split into cells
    col_positions = dot_cols.tolist()
    cell_splits = [0]
    for i in range(len(col_positions) - 1):
        if col_positions[i+1] - col_positions[i] > 15:
            cell_splits.append((col_positions[i] + col_positions[i+1]) // 2)
    cell_splits.append(w)

    result_letters = []
    result_confidences = []
    annotated = image_array.copy()

    for i in range(len(cell_splits) - 1):
        x_start = cell_splits[i]
        x_end = cell_splits[i+1]
        if x_end - x_start < 5:
            continue
        cell = image_array[:, x_start:x_end]
        if cell.size == 0:
            continue
        letter, confidence = predict_cell(cell)
        result_letters.append(letter)
        result_confidences.append(confidence)
        cv2.rectangle(annotated, (x_start, 0), (x_end, h), (0, 255, 0), 2)
        cv2.putText(annotated, letter.upper(), (x_start + 2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    sentence = "".join(result_letters)
    return sentence, result_confidences, annotated

def speak(text):
    tts = gTTS(text=text, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

st.title("⠃ BrailleVision")
st.caption("Physical Braille → English using Camera AI")
st.divider()

tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Camera"])

with tab1:
    uploaded = st.file_uploader("Upload a Braille image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Reading Braille..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)
        if sentence:
            st.image(annotated, caption="Detected cells", use_column_width=True)
            st.markdown("### 📝 Recognized Text")
            st.success(sentence.upper())
            if confidences:
                avg = sum(confidences) / len(confidences)
                st.metric("Average Confidence", f"{avg:.1f}%")
            if st.button("🔊 Read aloud"):
                audio = speak(sentence)
                st.audio(audio)
        else:
            st.warning("No Braille cells detected. Try a clearer image.")

with tab2:
    cam_img = st.camera_input("Point camera at Braille")
    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        img_array = np.array(img)
        with st.spinner("Reading Braille..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)
        if sentence:
            st.image(annotated, caption="Detected cells", use_column_width=True)
            st.markdown("### 📝 Recognized Text")
            st.success(sentence.upper())
            if confidences:
                avg = sum(confidences) / len(confidences)
                st.metric("Average Confidence", f"{avg:.1f}%")
            if st.button("🔊 Read aloud", key="cam_speak"):
                audio = speak(sentence)
                st.audio(audio)
        else:
            st.warning("No Braille cells detected. Try a clearer image.")
