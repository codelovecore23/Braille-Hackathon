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

st.set_page_config(page_title="BrailleVision", page_icon="⠃", layout="wide")

st.markdown("""
<style>
* { font-family: 'Segoe UI', sans-serif; }
.hero {
    background: linear-gradient(135deg, #007791 0%, #005f73 100%);
    padding: 50px 30px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 30px;
}
.hero h1 { color: white; font-size: 3.5rem; margin: 0; font-weight: 900; }
.hero p { color: rgba(255,255,255,0.85); font-size: 1.2rem; margin-top: 10px; }
.badge {
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.85rem;
    display: inline-block;
    margin-top: 12px;
    border: 1px solid rgba(255,255,255,0.3);
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 4px solid;
}
.step-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    height: 100%;
    border-left: 5px solid;
}
.tip-box {
    background: #E0F7FA;
    border-left: 4px solid #007791;
    border-radius: 10px;
    padding: 12px 16px;
    color: #005f73;
    font-size: 0.9rem;
    margin-bottom: 14px;
}
.footer {
    text-align: center;
    color: #999;
    font-size: 0.82rem;
    padding: 20px 0 10px 0;
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
        cv2.rectangle(annotated, (x_start, 0), (x_end, h), (0, 119, 145), 3)
        cv2.putText(annotated, letter.upper(), (x_start + 4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 119, 145), 2)
    sentence = "".join(result_letters)
    return sentence, result_confidences, annotated

def speak(text):
    tts = gTTS(text=text, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def show_results(sentence, confidences, annotated):
    # Result box
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #007791, #005f73);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,119,145,0.4);
        margin: 15px 0;
    ">
        <div style="font-size:0.9rem; opacity:0.85; margin-bottom:8px;">✅ Recognized Braille Text</div>
        <div style="font-size:4rem; font-weight:900; letter-spacing:12px;">{sentence.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence
    if confidences:
        avg = sum(confidences) / len(confidences)
        if avg > 90:
            bg, color, label = "#D1FAE5", "#065F46", f"🔥 {avg:.1f}% — Excellent"
        elif avg > 70:
            bg, color, label = "#FEF3C7", "#92400E", f"👍 {avg:.1f}% — Good"
        else:
            bg, color, label = "#FEE2E2", "#991B1B", f"⚠️ {avg:.1f}% — Low"
        st.markdown(f"""
        <div style="
            background:{bg};
            color:{color};
            border-radius:12px;
            padding:14px 20px;
            text-align:center;
            font-size:1rem;
            font-weight:600;
            margin:10px 0;
        ">
            🎯 Model Confidence: {label}
        </div>
        """, unsafe_allow_html=True)

    # Image and button
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(annotated, caption="Detected Cells", width=220)
    with col2:
        st.markdown("""
        <div style="padding-top:10px;">
            <div style="color:#007791; font-weight:700; font-size:1rem; margin-bottom:10px;">
                🔊 Text to Speech
            </div>
            <div style="color:#555; font-size:0.85rem; margin-bottom:12px;">
                Click below to hear the recognized Braille text read aloud
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶️ Read Aloud", use_container_width=True, type="primary"):
            with st.spinner("Generating audio..."):
                audio = speak(sentence)
                st.audio(audio)

# ── HERO ──
st.markdown("""
<div class="hero">
    <h1>⠃ BrailleVision</h1>
    <p>Real-time Physical Braille Recognition using Camera AI</p>
    <span class="badge">🏆 BrailleVision Hackathon 2026</span>
</div>
""", unsafe_allow_html=True)

# ── STATS ──
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="stat-card" style="border-color:#007791;">
        <div style="font-size:2.2rem;font-weight:900;color:#007791;">99%</div>
        <div style="color:#666;font-size:0.9rem;">🧠 Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-card" style="border-color:#005f73;">
        <div style="font-size:2.2rem;font-weight:900;color:#005f73;">26</div>
        <div style="color:#666;font-size:0.9rem;">🔤 Braille Classes (A–Z)</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-card" style="border-color:#0a9396;">
        <div style="font-size:2.2rem;font-weight:900;color:#0a9396;">Live</div>
        <div style="color:#666;font-size:0.9rem;">⚡ Real-time Camera</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── HOW IT WORKS ──
with st.expander("🔍 How BrailleVision Works", expanded=False):
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="step-card" style="border-color:#007791;">
            <div style="font-size:1.8rem;">📷</div>
            <div style="font-weight:700;color:#007791;margin:6px 0;">Step 1 — Capture</div>
            <div style="color:#555;font-size:0.88rem;">Upload a photo or use live camera to scan physical Braille</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card" style="border-color:#005f73;">
            <div style="font-size:1.8rem;">🔬</div>
            <div style="font-weight:700;color:#005f73;margin:6px 0;">Step 2 — Detect</div>
            <div style="color:#555;font-size:0.88rem;">OpenCV processes image, finds Braille dots and segments cells</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card" style="border-color:#0a9396;">
            <div style="font-size:1.8rem;">🧠</div>
            <div style="font-weight:700;color:#0a9396;margin:6px 0;">Step 3 — Predict</div>
            <div style="color:#555;font-size:0.88rem;">CNN model predicts each letter and reads text aloud</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🛠️ Tech Stack:** `Python` `TensorFlow` `OpenCV` `ONNX Runtime` `Streamlit` `gTTS`")

st.divider()

# ── TABS ──
tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Live Camera"])

with tab1:
    st.markdown('<div class="tip-box">💡 Upload a clear photo of physical or embossed Braille for best results</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a Braille image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="📄 Uploaded Image", width=220)
        with col2:
            with st.spinner("🔍 Analyzing Braille dots..."):
                sentence, confidences, annotated = find_and_predict_all(img_array)
            if sentence:
                st.success("✅ Braille detected successfully!")
            else:
                st.error("❌ No Braille detected")
        if sentence:
            st.divider()
            show_results(sentence, confidences, annotated)
        else:
            st.warning("⚠️ Try a clearer image with better lighting.")

with tab2:
    st.markdown('<div class="tip-box">📸 Point your camera at Braille text. Make sure lighting is bright and clear.</div>', unsafe_allow_html=True)
    cam_img = st.camera_input("📷 Scan Braille with Camera")
    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        img_array = np.array(img)
        with st.spinner("🔍 Analyzing Braille dots..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)
        if sentence:
            st.divider()
            show_results(sentence, confidences, annotated)
        else:
            st.warning("⚠️ No Braille detected. Try again with better lighting.")

# ── FOOTER ──
st.divider()
st.markdown("""
<div class="footer">
    Built with ❤️ for BrailleVision Hackathon 2026 &nbsp;|&nbsp;
    OpenCV + CNN + ONNX + Streamlit &nbsp;|&nbsp;
    Helping the visually impaired 👁️
</div>
""", unsafe_allow_html=True)
