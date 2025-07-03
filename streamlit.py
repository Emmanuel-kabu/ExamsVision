from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt 
from io import BytesIO
from datetime import datetime
import time
import seaborn as sns
import plotly.express as px
from PIL import Image
import pandas as pd

#Defining the classes for the Yolo model
classes_name ={
    0: "cheating",
    1: "good"
}

# Model path
model_path = "cheating-1/runs/detect/train22/weights/best.pt"
 #Load the YOLO model
model = YOLO(model_path)

# streamlt UI setup
st.set_page_config(
    page_title="ExamVisio", 
    layout="wide",
    initial_sidebar_state="expanded"
    )


# Sidebar - Navigation and Toggles
st.sidebar.title("Exam Visio Admin Panel")

page = st.sidebar.radio("📁 Navigate to:", [
    "Live Monitoring",
    "Analytics Dashboard",
    "Evidence Review",
    "Detection Settings",
    "Alerts",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Detection Toggles")

face_detection = st.sidebar.checkbox("Face Detection", value=True)
noise_detection = st.sidebar.checkbox("Noise Detection", value=True)
multi_face_detection = st.sidebar.checkbox("Multi-face Detection", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💼 Logged in as: `admin`")

# placeholder for the classes 
cheating_placeholder = st.sidebar.empty()
non_cheating_placeholder = st.sidebar.empty()
background_placeholder = st.sidebar.empty()


# Function to update class counts and display metrics
def update_class_counts(results, class_names):
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
    cv2.imwrite(filename, frame)
    st.write(f"Saved frame as: {filename}")


def obb_to_vertices(cx, cy, w, h, theta):
    """
    convert oriented bounding box parameterrs to vertices""" 
    theta_rad = np.radians(theta)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad), np.cos(theta_rad)]]) 
    
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ])

    vertices = np.dot(corners, R.T) + np.array([cx, cy])
    return vertices.astype(int)

def draw_obb_with_labels(image, vertices,  label, color, thickness=2):
    """
    Draw oriented bounding box with label on the image
    """
    vertices = vertices.reshape((-1, 1, 2))
    cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=thickness)
    top_vertices = tuple(vertices[vertices[:, :, 1].argmin()][0])
    label_pos = (top_vertices[0], top_vertices[1] - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    cv2.rectangle(image,
                  (label_pos[0], label_pos[1] - label_size[1] - 2), 
                  (label_pos[0] + label_size[0], label_pos[1] + 2), 
                  color, cv2.FILLED)
    
    cv2.putText(image, label, (label_pos[0], label_pos[1]), font, font_scale, (0, 0, 0), font_thickness,lineType=cv2.LINE_AA)
                   



# Example placeholders: You should already have these defined in your app
class_names = {0: "cheating", 1: "good"}  # update based on your YOLOv11x model
class_colors = {
    "good": (0, 255, 0),
    "cheating": (0, 0, 255)
}

def save_frame_with_timestamp(frame, confidence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"cheating_detections/cheating_{timestamp}_{confidence:.2f}.jpg"
    cv2.imwrite(filename, frame)
    st.write(f"Saved frame as: `{filename}`")


#process the video capture and display the results
def process_video_capture(video_capture, stframe, model):
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Run inference
        results = model(frame)
        if results is None:
            continue

        # Handle results (YOLOv11x-style)
        counts = update_class_counts(results, class_names)
        display_metrics(counts)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                confidence = float(box.conf)
                if confidence < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = class_names.get(class_id, 'Unknown')
                color = class_colors.get(class_name.lower(), (0, 255, 0))

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                # Save frame if cheating is detected
                if class_name.lower() == 'cheating' and confidence >= 0.85:
                    save_frame_with_timestamp(frame, confidence)

        # Convert and stream to Streamlit
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels='RGB')

    return st.button("Replay Video", key=f'replay-{time.time()}')



def live_monitoring():
    if page == "Live Monitoring":
        st.title("📹 Live Monitoring")
        st.title("Start Live Monitoring")
        source = st.selectbox(
    "Select Source", ("Webcam", "Video File")
)
    if source == "Webcam":
        model = model.track()  # Load the YOLO model
        video_capture  = cv2.VideoCapture(0)  # Use webcam
        stframe = st.empty()
        process_video_capture(video_capture, stframe, model)
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
                replay_button = process_video_capture(video_capture, stframe, model)
                video_capture.release()
                cv2.destroyAllWindows()   
                return replay_button
           while True:
                replay= process_uploaded_video()
                if not replay:
                    break
                


# Visualiing the output of the model and inpecting the results for evidence review
def visualize_results(results, frame, page):
    if page != "Analytics Dashboard":
        return

    st.title("📊 Analytics Dashboard")
    st.subheader("Session Overview & Behavioral Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart", "Line Chart", "Heatmap", "Pie Chart"])

    st.write("Visualizing the results after the exam session...")

    counts = update_class_counts(results, class_names)
    display_metrics(counts)

    # Fallback in case keys are missing
    cheating = counts.get('cheating', 0)
    good = counts.get('good', 0)

    with tab1:
        st.metric("Total Sessions Reviewed", cheating + good)
        st.bar_chart(pd.DataFrame({
            'Label': ['Cheating', 'Good'],
            'Count': [cheating, good]
        }).set_index('Label'))

    with tab2:
        st.line_chart(pd.DataFrame({
            'Cheating': [cheating],
            'Good': [good]
        }))

    with tab3:
        data = pd.DataFrame({
            'Cheating': [cheating],
            'Good': [good]
        })
        fig, ax = plt.subplots()
        sns.heatmap(data, annot=True, cmap="coolwarm", fmt='d', ax=ax)
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots()
        ax.pie([cheating, good], labels=['Cheating', 'Good'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)




def display_captured_images(folder="cheating_detections"):
    st.title("📸 Captured Cheating Detections")

    if not os.path.exists(folder):
        st.warning("No detection folder found yet.")
        return

    files = sorted(os.listdir(folder), reverse=True)  # latest first
    image_files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]

    if not image_files:
        st.info("No images captured yet.")
        return

    for file in image_files:
        image_path = os.path.join(folder, file)
        image = Image.open(image_path)

        # Show image with caption
        st.image(image, caption=f"{file}", use_column_width=True)
        st.write(f"Captured at: {file.split('_')[1]}")  # Assuming filename format includes timestamp



def alerts():
    st.title("🚨 Alerts & Notifications")
    st.write("This section will display real-time alerts and notifications related to detected cheating activities.")
    # Placeholder for future implementation
    if class_names.get(0) == "cheating":
        st.warning("Cheating detected! Please review the captured images in the Evidence Review section.")



