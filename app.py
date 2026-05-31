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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 40px 20px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: 2px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #90CAF9;
        margin-top: 8px;
    }
    .hero-badge {
        display: inline-block;
        background: #1E88E5;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-top: 10px;
    }
    .result-box {
        background: linear-gradient(135deg, #1565C0, #1E88E5, #42A5F5);
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        letter-spacing: 8px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(30,136,229,0.4);
    }
    .confidence-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        padding: 12px 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1rem;
        color: #2e7d32;
        font-weight: 600;
        margin: 8px 0;
        border-left: 4px solid #43A047;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #e8eaf6);
        padding: 14px 18px;
        border-radius: 12px;
        font-size: 0.9rem;
        color: #1565C0;
        margin: 8px 0;
        border-left: 4px solid #1E88E5;
    }
    .step-box {
        background: #1a1a2e;
        padding: 15px 20px;
        border-radius: 12px;
        color: #e0e0e0;
        margin: 5px 0;
        font-size: 0.9rem;
        border-left: 4px solid #1E88E5;
    }
    .tech-badge {
        display: inline-block;
        background: #1a237e;
        color: #90CAF9;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 3px;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 20px;
        padding: 15px;
        border-top: 1px solid #eee;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
        border-radius: 8px;
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
    st.markdown(f'<div class="result-box">⠿ {sentence.upper()}</div>', unsafe_allow_html=True)
    if confidences:
        avg = sum(confidences) / len(confidences)
        st.markdown(f'<div class="confidence-box">🎯 Model Confidence: {avg:.1f}% — {"Excellent" if avg > 90 else "Good" if avg > 70 else "Low"}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(annotated, caption="🟢 Detected Cells", width=240)
    with col2:
        st.markdown("#### 🔊 Listen")
        st.markdown("Click below to hear the recognized text read aloud:")
        if st.button("▶️ Read Aloud", use_container_width=True, type="primary"):
            with st.spinner("Generating audio..."):
                audio = speak(sentence)
                st.audio(audio)
        st.markdown("#### 📋 Copy Text")
        st.code(sentence.upper(), language=None)

# ── HERO SECTION ──
st.markdown("""
<div class="hero">
    <div class="hero-title">⠃ BrailleVision</div>
    <div class="hero-subtitle">Real-time Physical Braille Recognition using Camera AI</div>
    <div class="hero-badge">🏆 BrailleVision Hackathon 2026</div>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ──
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🧠 Model Accuracy", "99%", "CNN")
with col2:
    st.metric("🔤 Classes", "26", "A to Z")
with col3:
    st.metric("⚡ Processing", "Real-time", "Fast")

st.divider()

# ── HOW IT WORKS ──
with st.expander("🔍 How BrailleVision Works", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="step-box">📷 <b>Step 1</b><br>Upload or capture a Braille image using camera</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-box">🔬 <b>Step 2</b><br>OpenCV detects and segments Braille dot cells</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-box">🧠 <b>Step 3</b><br>CNN predicts each letter → Text + Speech output</div>', unsafe_allow_html=True)

    st.markdown("#### 🛠️ Tech Stack")
    st.markdown("""
    <span class="tech-badge">Python</span>
    <span class="tech-badge">TensorFlow</span>
    <span class="tech-badge">OpenCV</span>
    <span class="tech-badge">ONNX Runtime</span>
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">gTTS</span>
    """, unsafe_allow_html=True)

st.divider()

# ── MAIN TABS ──
tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Live Camera"])

with tab1:
    st.markdown('<div class="info-box">💡 Upload a clear photo of physical or embossed Braille paper for best results</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a Braille image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="📄 Uploaded Image", width=240)
        with col2:
            st.markdown("#### ⚙️ Processing")
            with st.spinner("🔍 Analyzing Braille dots..."):
                sentence, confidences, annotated = find_and_predict_all(img_array)
            if sentence:
                st.success("✅ Braille detected successfully!")
            else:
                st.error("❌ No Braille detected")
        if sentence:
            st.divider()
            st.markdown("### 📝 Recognition Result")
            show_results(sentence, confidences, annotated)
        else:
            st.warning("⚠️ No Braille cells detected. Try a clearer image with better lighting.")

with tab2:
    st.markdown('<div class="info-box">📸 Point your camera directly at the Braille text. Make sure lighting is good.</div>', unsafe_allow_html=True)
    cam_img = st.camera_input("📷 Scan Braille with Camera")
    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        img_array = np.array(img)
        with st.spinner("🔍 Analyzing Braille dots..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)
        if sentence:
            st.divider()
            st.markdown("### 📝 Recognition Result")
            show_results(sentence, confidences, annotated)
        else:
            st.warning("⚠️ No Braille cells detected. Try again with better lighting.")

# ── FOOTER ──
st.divider()
st.markdown("""
<div class="footer">
    Built with ❤️ for BrailleVision Hackathon 2026 &nbsp;|&nbsp;
    OpenCV + CNN + ONNX + Streamlit &nbsp;|&nbsp;
    Helping the visually impaired 👁️
</div>
""", unsafe_allow_html=True)
