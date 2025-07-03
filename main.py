from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import seaborn as sns
from PIL import Image
import pandas as pd

# Define the classes for the YOLO model
class_names = {0: "cheating", 1: "good"}
class_colors = {"cheating": (0, 0, 255), "good": (0, 255, 0)}

# Load the YOLO model
model_path = "cheating-1/runs/detect/train22/weights/best.pt"
model = YOLO(model_path)

# Streamlit page setup
st.set_page_config(page_title="ExamVisio", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("Exam Visio Admin Panel")
page = st.sidebar.radio("\U0001F4C1 Navigate to:", [
    "Live Monitoring",
    "Analytics Dashboard",
    "Evidence Review",
    "Detection Settings",
    "Alerts",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### \u2699\ufe0f Detection Toggles")
face_detection = st.sidebar.checkbox("Face Detection", value=True)
noise_detection = st.sidebar.checkbox("Noise Detection", value=True)
multi_face_detection = st.sidebar.checkbox("Multi-face Detection", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("\U0001F9D1‍\U0001F4BC Logged in as: `admin`")

# Sidebar metrics placeholders
cheating_placeholder = st.sidebar.empty()
non_cheating_placeholder = st.sidebar.empty()

# Update and display detection metrics
def update_class_counts(results):
    counts = {'cheating': 0, 'good': 0}
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = class_names.get(class_id, 'Unknown')
            if class_name.lower() == 'cheating':
                counts['cheating'] += 1
            else:
                counts['good'] += 1
    return counts

def display_metrics(counts):
    cheating_placeholder.metric(label='cheating', value=counts['cheating'])
    non_cheating_placeholder.metric(label='good', value=counts['good'])

def save_frame_with_timestamp(frame, confidence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"cheating_detections/cheating_{timestamp}_{confidence:.2f}.jpg"
    os.makedirs("cheating_detections", exist_ok=True)
    cv2.imwrite(filename, frame)
    st.write(f"Saved frame as: `{filename}`")

def process_video_capture(video_capture, stframe):
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        results = model.track(frame)
        if results is None:
            continue

        counts = update_class_counts(results)
        display_metrics(counts)

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = class_names.get(class_id, 'Unknown')
                color = class_colors.get(class_name.lower(), (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_name.lower() == 'cheating' and confidence >= 0.85:
                    save_frame_with_timestamp(frame, confidence)

        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels='RGB')

    return st.button("Replay Video", key=f'replay-{time.time()}')

def live_monitoring():
    st.title("\U0001F4F9 Live Monitoring")
    source = st.selectbox("Select Source", ("Webcam", "Video File"))

    if source == "Webcam":
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        process_video_capture(video_capture, stframe)
        video_capture.release()
        cv2.destroyAllWindows()

    elif source == "Video File":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmpfile_path = tmpfile.name

            stframe = st.empty()
            def process_uploaded_video():
                video_capture = cv2.VideoCapture(tmpfile_path)
                replay_button = process_video_capture(video_capture, stframe)
                video_capture.release()
                cv2.destroyAllWindows()
                return replay_button

            while True:
                if not process_uploaded_video():
                    break

def visualize_results():
    st.title("\U0001F4CA Analytics Dashboard")
    st.subheader("Session Overview & Behavioral Analytics")

    # Dummy data (replace with real inference results)
    cheating, good = 10, 90

    tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart", "Line Chart", "Heatmap", "Pie Chart"])

    with tab1:
        st.bar_chart(pd.DataFrame({'Label': ['Cheating', 'Good'], 'Count': [cheating, good]}).set_index('Label'))

    with tab2:
        st.line_chart(pd.DataFrame({'Cheating': [cheating], 'Good': [good]}))

    with tab3:
        data = pd.DataFrame({'Cheating': [cheating], 'Good': [good]})
        fig, ax = plt.subplots()
        sns.heatmap(data, annot=True, cmap="coolwarm", fmt='d', ax=ax)
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots()
        ax.pie([cheating, good], labels=['Cheating', 'Good'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

def display_captured_images(folder="cheating_detections"):
    st.title("\U0001F4F8 Captured Cheating Detections")
    if not os.path.exists(folder):
        st.warning("No detection folder found yet.")
        return
    files = sorted(os.listdir(folder), reverse=True)
    image_files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]
    if not image_files:
        st.info("No images captured yet.")
        return
    for file in image_files:
        image_path = os.path.join(folder, file)
        image = Image.open(image_path)
        st.image(image, caption=f"{file}", use_column_width=True)
        st.write(f"Captured at: {file.split('_')[1]}")

def alerts():
    st.title("\U0001F6A8 Alerts & Notifications")
    st.warning("Cheating detected! Please review the Evidence Review section.")

# Routing
if page == "Live Monitoring":
    live_monitoring()
elif page == "Analytics Dashboard":
    visualize_results()
elif page == "Evidence Review":
    display_captured_images()
elif page == "Alerts":
    alerts()
elif page == "Detection Settings":
    st.info("Detection toggles are configurable in the sidebar above.")
