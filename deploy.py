import streamlit as st
import cv2
import time
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import geocoder

# -------------------------------
# Load YOLOv12 model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("./model/best.pt")

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Trash Detection", layout="wide")

st.title("🗑️ Trash Detection on Mountain Hiking Trail")
st.markdown("Real-time detection using your camera. Objects are counted every few seconds.")

# Sidebar controls
st.sidebar.header("⚙️ Options")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
interval = st.sidebar.slider("Capture Interval (seconds)", 1, 10, 3)

# -------------------------------
# Initialize session state
# -------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "data" not in st.session_state:
    st.session_state.data = []
if "last_capture_time" not in st.session_state:
    st.session_state.last_capture_time = 0
if "cumulative_counts" not in st.session_state:
    st.session_state.cumulative_counts = {}

# -------------------------------
# Start/Stop/Pause buttons
# -------------------------------
col1, col2, col3 = st.columns(3)

if col1.button("▶️ Start Detection"):
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.data = []
    st.session_state.cumulative_counts = {}
    st.session_state.last_capture_time = 0

if col2.button("⏸️ Pause/Resume Detection"):
    if st.session_state.running:
        st.session_state.paused = not st.session_state.paused

if col3.button("⏹️ Stop Detection"):
    st.session_state.running = False
    st.session_state.paused = False

# -------------------------------
# Detection
# -------------------------------
frame_window = st.empty()

if st.session_state.running:
    cap = cv2.VideoCapture(0)  # webcam

    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        annotated_frame = frame.copy()

        # Always run detection → for bounding boxes
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        annotated_frame = results[0].plot()  # bounding boxes on frame

        if not st.session_state.paused:
            # Log results only every `interval` seconds
            if current_time - st.session_state.last_capture_time >= interval:
                st.session_state.last_capture_time = current_time

                names = model.names
                counts = {}
                labels_detected = []
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = names[cls]
                    labels_detected.append(label)
                    counts[label] = counts.get(label, 0) + 1
                    st.session_state.cumulative_counts[label] = (
                        st.session_state.cumulative_counts.get(label, 0) + 1
                    )

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                geo = geocoder.ip("me").latlng

                st.session_state.data.append({
                    "timestamp": timestamp,
                    "geo": geo,
                    "labels": ", ".join(labels_detected) if labels_detected else None,
                    "counts": counts
                })

        # Show live frame with bounding boxes
        frame_window.image(annotated_frame, channels="BGR", use_container_width=True)

        time.sleep(0.03)

    cap.release()

# -------------------------------
# Results Table + Save Option
# -------------------------------
if st.session_state.data:
    st.subheader("📊 Detection Log")

    df = pd.DataFrame(st.session_state.data)
    st.dataframe(df)

    st.markdown("### 🔢 Total Counts Since Start")
    st.json(st.session_state.cumulative_counts)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Data as CSV", csv, "detections.csv", "text/csv")
