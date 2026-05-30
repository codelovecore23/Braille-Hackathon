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
        gdown.download("https://drive.google.com/uc?id=12CUhIdfRJqJQkuxVQ0WgCbEFctRAe1P4", "braille_cnn.onnx", quiet=False)
    if not os.path.exists("reverse_map.json"):
        gdown.download("https://drive.google.com/uc?id=1kkeMXthfkod3D7rxHk0NJGUad6-__vhs", "reverse_map.json", quiet=False)
    session = ort.InferenceSession("braille_cnn.onnx")
    with open("reverse_map.json") as f:
        reverse_map = json.load(f)
    return session, reverse_map

session, reverse_map = load_resources()

def predict_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Tight bounding box around dots only
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        xb, yb, wb, hb = cv2.boundingRect(coords)
        pad = 5
        xb = max(0, xb - pad)
        yb = max(0, yb - pad)
        wb = min(gray.shape[1] - xb, wb + 2*pad)
        hb = min(gray.shape[0] - yb, hb + 2*pad)
        gray = gray[yb:yb+hb, xb:xb+wb]

    # Pad to square before resize
    h, w = gray.shape
    size = max(h, w)
    square = np.ones((size, size), dtype=np.uint8) * 255
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = gray

    resized = cv2.resize(square, (50, 50))
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

    # Crop top 50% to remove printed text at bottom
    cropped = image_array[:int(h * 0.50), :]

    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect dot centers
    dot_centers = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if 20 < area < 600:
            cx = x + cw // 2
            cy = y + ch // 2
            dot_centers.append((cx, cy))

    if not dot_centers:
        return "", [], cropped

    # Step 1: Cluster dot x-positions into columns
    xs = sorted([d[0] for d in dot_centers])

    col_boundaries = []
    for i in range(len(xs) - 1):
        if xs[i+1] - xs[i] > 10:
            col_boundaries.append((xs[i] + xs[i+1]) // 2)

    col_groups = []
    prev = 0
    for b in col_boundaries:
        col_x = [x for x in xs if prev <= x < b]
        if col_x:
            col_groups.append(int(np.mean(col_x)))
        prev = b
    last_col = [x for x in xs if x >= prev]
    if last_col:
        col_groups.append(int(np.mean(last_col)))

    if len(col_groups) < 2:
        return "", [], cropped

    # Step 2: Calculate gaps between columns
    col_gaps = [col_groups[i+1] - col_groups[i] for i in range(len(col_groups) - 1)]

    # Step 3: Fixed threshold — small gap = same cell, large gap = new cell
    # From your debug: small gaps ~22px, large gaps ~60px
    # So 35px is a clean threshold between them
    SAME_CELL_THRESHOLD = 35

    cell_col_groups = []
    i = 0
    while i < len(col_groups):
        if i + 1 < len(col_groups):
            gap = col_groups[i+1] - col_groups[i]
            if gap < SAME_CELL_THRESHOLD:
                # Two columns = one Braille letter
                cell_col_groups.append((col_groups[i], col_groups[i+1]))
                i += 2
            else:
                # Single column letter (like 'I' or 'A')
                cell_col_groups.append((col_groups[i], col_groups[i]))
                i += 1
        else:
            cell_col_groups.append((col_groups[i], col_groups[i]))
            i += 1

    # Step 4: Crop and predict each cell
    result_letters = []
    result_confidences = []
    annotated = cropped.copy()
    padding = 20

    for (lx, rx) in cell_col_groups:
        x_start = max(0, lx - padding)
        x_end = min(cropped.shape[1], rx + padding)
        if x_end - x_start < 10:
            continue
        cell = cropped[:, x_start:x_end]
        if cell.size == 0:
            continue
        letter, confidence = predict_cell(cell)
        result_letters.append(letter)
        result_confidences.append(confidence)
        cv2.rectangle(annotated, (x_start, 0), (x_end, cropped.shape[0]), (0, 255, 0), 2)
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
