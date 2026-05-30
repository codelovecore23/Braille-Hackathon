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

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #888;
        margin-bottom: 20px;
    }
    .result-box {
        background: linear-gradient(135deg, #1E88E5, #42A5F5);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: white;
        letter-spacing: 6px;
        margin: 10px 0;
    }
    .confidence-box {
        background: #f0f4ff;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 1rem;
        color: #333;
        margin: 5px 0;
    }
    .info-box {
        background: #e8f5e9;
        padding: 12px;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #2e7d32;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

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
    col_positions = dot_cols.tolist()
    cell_splits = [0]
    for i in range(len(col_positions) - 1):
        if col_positions[i+1] - col_positions[i] > 80:
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

def show_results(sentence, confidences, annotated):
    st.markdown(f'<div class="result-box">{sentence.upper()}</div>', unsafe_allow_html=True)
    if confidences:
        avg = sum(confidences) / len(confidences)
        st.markdown(f'<div class="confidence-box">🎯 Average Confidence: <b>{avg:.1f}%</b></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated, caption="Detected cells", width=250)
    if st.button("🔊 Read aloud", use_container_width=True):
        audio = speak(sentence)
        st.audio(audio)

# ── Header ──
st.markdown('<div class="main-title">⠃ BrailleVision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Physical Braille → English using Camera AI</div>', unsafe_allow_html=True)
st.divider()

# ── How to use ──
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Upload a **clear photo** of physical Braille
    2. Or use your **camera** to scan
    3. App detects Braille dots and predicts letters
    4. Click **Read aloud** to hear the result
    """)

st.divider()

tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Camera"])

with tab1:
    uploaded = st.file_uploader("Upload a Braille image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", width=250)

        with st.spinner("🔍 Reading Braille..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)

        if sentence:
            st.markdown("### 📝 Recognized Text")
            show_results(sentence, confidences, annotated)
        else:
            st.warning("⚠️ No Braille cells detected. Try a clearer image.")

with tab2:
    st.markdown('<div class="info-box">📸 Point your camera directly at the Braille text for best results</div>', unsafe_allow_html=True)
    cam_img = st.camera_input("Scan Braille")
    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        img_array = np.array(img)

        with st.spinner("🔍 Reading Braille..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)

        if sentence:
            st.markdown("### 📝 Recognized Text")
            show_results(sentence, confidences, annotated)
        else:
            st.warning("⚠️ No Braille cells detected. Try a clearer image.")

st.divider()
st.markdown('<div class="sub-title">Built with ❤️ using OpenCV + CNN + Streamlit</div>', unsafe_allow_html=True)
