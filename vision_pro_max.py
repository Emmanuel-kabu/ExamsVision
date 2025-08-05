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
import torch
import threading
from queue import Queue
import yaml
from yaml.loader import SafeLoader
import webbrowser
from oauthlib.oauth2 import WebApplicationClient
import requests
import smtplib
import streamlit_authenticator as stauth
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import json

# Configuration
GOOGLE_CLIENT_ID = "your-google-client-id.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "your-google-client-secret"
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Define the classes for the YOLO model
class_names = {0: "cheating", 1: "good"}
class_colors = {"cheating": (0, 0, 255), "good": (0, 255, 0)}

# Load the YOLO model
model_path = "runs/detect/train22/weights/best.pt"
model = YOLO(model_path)

# Initialize session state
# Initialize session state
def init_session_state():
    if 'detection_counts' not in st.session_state:
        st.session_state.detection_counts = {'cheating': 0, 'good': 0}
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = pd.DataFrame(columns=['timestamp', 'cheating', 'good'])
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'frame_size' not in st.session_state:
        st.session_state.frame_size = (640, 480)
    if 'video_queue' not in st.session_state:
        st.session_state.video_queue = Queue(maxsize=10)
    if 'processing_thread' not in st.session_state:
        st.session_state.processing_thread = None
    # Authentication states
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'email_alerts' not in st.session_state:
        st.session_state.email_alerts = False
    if 'name' not in st.session_state:
        st.session_state.name = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'authenticator' not in st.session_state:
        st.session_state.authenticator = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Live Monitoring"

init_session_state()

# Authentication Functions
def get_google_auth():
    client = WebApplicationClient(GOOGLE_CLIENT_ID)
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=st.experimental_get_query_params().get("redirect_uri", ["http://localhost:8501"])[0],
        scope=["openid", "email", "profile"],
    )
    return request_uri
    

# Video Processing Functions
def video_capture_thread(video_source=0):
    try:
        cap = cv2.VideoCapture(video_source)
        while st.session_state.get('monitoring_active', False):
            ret, frame = cap.read()
            if not ret:
                break
            if st.session_state.video_queue.full():
                st.session_state.video_queue.get()
            st.session_state.video_queue.put(frame)
    except Exception as e:
        st.error(f"Video capture error: {str(e)}")
    finally:
        cap.release()

def start_monitoring_thread(video_source=0):
    st.session_state.monitoring_active = True
    st.session_state.processing_thread = threading.Thread(
        target=video_capture_thread,
        args=(video_source,),
        daemon=True
    )
    st.session_state.processing_thread.start()

def stop_monitoring():
    st.session_state.monitoring_active = False
    if st.session_state.processing_thread is not None:
        st.session_state.processing_thread.join(timeout=1)
    st.session_state.video_queue = Queue(maxsize=10)
    st.experimental_rerun()

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
    
    st.session_state.detection_counts['cheating'] += counts['cheating']
    st.session_state.detection_counts['good'] += counts['good']
    
    if counts['cheating'] > 0 or counts['good'] > 0:
        new_entry = {
            'timestamp': datetime.now(),
            'cheating': counts['cheating'],
            'good': counts['good']
        }
        st.session_state.detection_history = pd.concat([
            st.session_state.detection_history,
            pd.DataFrame([new_entry])
        ], ignore_index=True)
    
    return counts


def display_metrics(counts, person_count=0):
    cheating_placeholder.metric(label='Cheating Detections', value=st.session_state.detection_counts['cheating'])
    non_cheating_placeholder.metric(label='Good Behavior', value=st.session_state.detection_counts['good'])
    alert_placeholder.metric(label='Active Alerts', value=len(st.session_state.alert_history))
    person_count_placeholder.metric(label='People in Frame', value=person_count)

def send_email_alert(alert_data):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_ADDRESS = "your-email@gmail.com"
    EMAIL_PASSWORD = "your-app-password"
    RECIPIENT_EMAIL = "recipient@example.com"
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = f"ExamVisio Pro Alert - Cheating Detected ({alert_data['confidence']:.2f} confidence)"
    
    body = f"""
    <h2>Cheating Incident Detected</h2>
    <p><strong>Time:</strong> {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Confidence:</strong> {alert_data['confidence']:.2f}</p>
    <p><strong>People in frame:</strong> {alert_data['people_count']}</p>
    <p>See attached image for details.</p>
    """
    msg.attach(MIMEText(body, 'html'))
    
    with open(alert_data['image_path'], 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(alert_data['image_path']))
        msg.attach(img)
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def alert_notification(alert_data):
    js = f"""
    <script>
        if (Notification.permission === "granted") {{
            new Notification("Cheating Detected", {{
                body: "Confidence: {alert_data['confidence']:.2f}\\nTime: {alert_data['timestamp'].strftime('%H:%M:%S')}",
                icon: "https://cdn-icons-png.flaticon.com/512/1828/1828640.png"
            }});
        }} else if (Notification.permission !== "denied") {{
            Notification.requestPermission().then(permission => {{
                if (permission === "granted") {{
                    new Notification("Cheating Detected", {{
                        body: "Confidence: {alert_data['confidence']:.2f}\\nTime: {alert_data['timestamp'].strftime('%H:%M:%S')}",
                        icon: "https://cdn-icons-png.flaticon.com/512/1828/1828640.png"
                    }});
                }}
            }});
        }}
    </script>
    """
    st.components.v1.html(js, height=0)

def save_frame_with_timestamp(frame, confidence, boxes):
    timestamp = datetime.now()
    filename = f"cheating_detections/cheating_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{confidence:.2f}.jpg"
    os.makedirs("cheating_detections", exist_ok=True)
    
    annotated_frame = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls)
        class_name = class_names.get(class_id, 'Unknown')
        color = class_colors.get(class_name.lower(), (0, 255, 0))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        label = f'{class_name} {float(box.conf):.2f}'
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(filename, annotated_frame)
    
    alert_data = {
        'timestamp': timestamp,
        'image_path': filename,
        'confidence': confidence,
        'people_count': len(boxes)
    }
    st.session_state.alert_history.append(alert_data)
    
    alert_notification(alert_data)
    
    if st.session_state.email_alerts:
        threading.Thread(
            target=send_email_alert,
            args=(alert_data,),
            daemon=True
        ).start()

def process_frames(stframe):
    last_update_time = time.time()
    last_alert_time = 0
    alert_cooldown = 5 
    confidence_threshold = 0.5 # seconds between alerts
    
    while st.session_state.monitoring_active:
        if not st.session_state.video_queue.empty():
            frame = st.session_state.video_queue.get()
            frame = cv2.resize(frame, st.session_state.frame_size)
            
            results = model.predict(
                frame,
                conf=confidence_threshold,
                iou=0.5,
                imgsz=640,
                device=torch.cuda.is_available() and 'cuda' or 'cpu',
            )
            
            person_count = sum(len(result.boxes) for result in results)
            counts = update_class_counts(results)
            display_metrics(counts, person_count)

            cheating_boxes = []
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = class_names.get(class_id, 'Unknown')
                    color = class_colors.get(class_name.lower(), (0, 255, 0))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if class_name.lower() == 'cheating' and confidence >= 0.60:
                        cheating_boxes.append(box)

            current_time = time.time()
            if cheating_boxes and (current_time - last_alert_time) > alert_cooldown:
                max_confidence = max(float(box.conf) for box in cheating_boxes)
                save_frame_with_timestamp(frame, max_confidence, cheating_boxes)
                last_alert_time = current_time

            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels='RGB')

            if time.time() - last_update_time > 2 and st.session_state.current_page == "Analytics Dashboard":
                last_update_time = time.time()
                st.rerun()

# Page Functions
def live_monitoring():
    st.title("\U0001F4F9 Live Monitoring Dashboard")
    source = st.selectbox("Select Source", ("Webcam", "Video File"))

    if source == "Webcam":
        stframe = st.empty()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Monitoring", key="start_webcam"):
                start_monitoring_thread(0)
                process_frames(stframe)
        with col2:
            if st.button("Stop Monitoring", key="stop_webcam"):
                stop_monitoring()

    elif source == "Video File":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmpfile_path = tmpfile.name

            stframe = st.empty()
            start_monitoring_thread(tmpfile_path)
            process_frames(stframe)
            
            if st.button("Stop Monitoring", key="stop_video"):
                stop_monitoring()
                os.unlink(tmpfile_path)

def visualize_results():
    st.title("\U0001F4CA Analytics Dashboard")
    
    if not st.session_state.monitoring_active and st.session_state.detection_history.empty:
        st.info("""
        **No monitoring data available yet!**
        
        To view analytics:
        1. Go to **Live Monitoring**
        2. Start monitoring using either:
           - Your webcam
           - An uploaded video file
        3. Return here to view real-time analytics
        """)
        st.image("https://via.placeholder.com/800x400?text=Start+Monitoring+to+See+Analytics", use_container_width=True)
        return
    
    st.subheader("Behavioral Analytics Overview")

    cheating = st.session_state.detection_counts['cheating']
    good = st.session_state.detection_counts['good']

    time_window = st.selectbox("Time Window", ["Last 5 minutes", "Last 15 minutes", "Last hour", "All time"])
    
    now = datetime.now()
    if time_window == "Last 5 minutes":
        filtered_history = st.session_state.detection_history[st.session_state.detection_history['timestamp'] > now - pd.Timedelta(minutes=5)]
    elif time_window == "Last 15 minutes":
        filtered_history = st.session_state.detection_history[st.session_state.detection_history['timestamp'] > now - pd.Timedelta(minutes=15)]
    elif time_window == "Last hour":
        filtered_history = st.session_state.detection_history[st.session_state.detection_history['timestamp'] > now - pd.Timedelta(hours=1)]
    else:
        filtered_history = st.session_state.detection_history

    tab1, tab2, tab3, tab4 = st.tabs(["Behavior Distribution", "Timeline Analysis", "Detection Heatmap", "Proportions"])

    with tab1:
        if cheating == 0 and good == 0:
            st.warning("No behavior data detected yet")
        else:
            fig, ax = plt.subplots()
            sns.barplot(x=['Cheating', 'Good'], y=[cheating, good], palette=['red', 'green'], ax=ax)
            ax.set_title('Behavior Detection Counts')
            ax.set_ylabel('Number of Detections')
            st.pyplot(fig)

    with tab2:
        if filtered_history.empty:
            st.warning("No detection data available for the selected time window")
        else:
            resampled = filtered_history.set_index('timestamp').resample('30S').sum()
            if resampled.empty:
                st.warning("Not enough data points to display timeline")
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                resampled.plot(ax=ax)
                ax.set_title('Detection Timeline')
                ax.set_ylabel('Detections per 30 Seconds')
                st.pyplot(fig)

    with tab3:
        if cheating == 0 and good == 0:
            st.warning("No data available for heatmap visualization")
        else:
            data = pd.DataFrame({'Cheating': [cheating], 'Good': [good]})
            fig, ax = plt.subplots()
            sns.heatmap(data, annot=True, cmap="coolwarm", fmt='d', ax=ax)
            ax.set_title('Detection Intensity')
            st.pyplot(fig)

    with tab4:
        if cheating == 0 and good == 0:
            st.warning("No behavior data available for pie chart")
        else:
            fig, ax = plt.subplots()
            ax.pie([cheating, good], labels=['Cheating', 'Good'], autopct='%1.1f%%', 
                   startangle=90, colors=['red', 'green'], explode=(0.1, 0))
            ax.axis('equal')
            ax.set_title('Behavior Proportions')
            st.pyplot(fig)

def display_captured_images():
    st.title("\U0001F4F8 Evidence Gallery")
    if not os.path.exists("cheating_detections"):
        st.warning("No detection folder found yet.")
        st.info("Potential cheating incidents will appear here once detected during monitoring")
        return
    
    st.subheader("Detection Timeline")
    if not st.session_state.alert_history:
        st.success("No cheating incidents detected yet!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.6, 0.05)
        with col2:
            min_people = st.slider("Minimum People in Frame", 1, 10, 1)
        
        filtered_alerts = [
            alert for alert in st.session_state.alert_history
            if alert['confidence'] >= min_confidence and alert['people_count'] >= min_people
        ]
        
        if not filtered_alerts:
            st.warning("No alerts match your filters")
        else:
            st.write(f"Showing {len(filtered_alerts)} alerts matching your criteria")
            for alert in reversed(filtered_alerts):
                expander = st.expander(
                    f"Alert at {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"Confidence: {alert['confidence']:.2f} - "
                    f"People: {alert['people_count']}"
                )
                with expander:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if os.path.exists(alert['image_path']):
                            st.image(alert['image_path'], use_container_width=True)
                    with col2:
                        st.write(f"**Detection Details:**")
                        st.write(f"- **Timestamp:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"- **Confidence Level:** {alert['confidence']:.2f}")
                        st.write(f"- **People Detected:** {alert['people_count']}")
                        st.download_button(
                            label="Download Evidence",
                            data=open(alert['image_path'], "rb").read(),
                            file_name=os.path.basename(alert['image_path']),
                            key=f"dl_{alert['timestamp'].timestamp()}"
                        )

def alerts():
    st.title("\U0001F6A8 Alert Management")
    
    if not st.session_state.alert_history:
        st.success("No alerts generated - no cheating detected!")
    else:
        st.warning(f"**{len(st.session_state.alert_history)} cheating incidents detected!**")
        
        st.subheader("Alert Filters")
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("Filter by date", [])
        with col2:
            severity_filter = st.select_slider("Filter by confidence", options=['Low', 'Medium', 'High'], value='Low')
        
        min_conf = 0.5 if severity_filter == 'Low' else 0.7 if severity_filter == 'Medium' else 0.85
        
        filtered_alerts = [
            alert for alert in st.session_state.alert_history
            if alert['confidence'] >= min_conf and 
            (not date_filter or alert['timestamp'].date() in date_filter)
        ]
        
        if not filtered_alerts:
            st.warning("No alerts match your filters")
        else:
            st.write(f"Showing {len(filtered_alerts)} filtered alerts")
            
            for i, alert in enumerate(reversed(filtered_alerts), 1):
                alert_card = st.container()
                with alert_card:
                    cols = st.columns([1, 4, 1])
                    with cols[0]:
                        if os.path.exists(alert['image_path']):
                            st.image(alert['image_path'], width=150)
                    with cols[1]:
                        st.subheader(f"Alert #{i}")
                        st.write(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Confidence:** {alert['confidence']:.2f}")
                        st.write(f"**People in Frame:** {alert['people_count']}")
                    with cols[2]:
                        st.download_button(
                            "Download",
                            data=open(alert['image_path'], "rb").read(),
                            file_name=f"evidence_{i}.jpg",
                            key=f"alert_dl_{i}"
                        )
                st.markdown("---")

def history_page():
    st.title("📜 Monitoring History")
    
    if st.session_state.detection_history.empty:
        st.info("No monitoring history available yet.")
        return
    
    st.subheader("Session Summary")
    
    total_sessions = len(st.session_state.detection_history['timestamp'].dt.date.unique())
    total_alerts = len(st.session_state.alert_history)
    avg_cheating = st.session_state.detection_history['cheating'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", total_sessions)
    col2.metric("Total Alerts", total_alerts)
    col3.metric("Avg Cheating per Session", f"{avg_cheating:.1f}")
    
    st.subheader("Detailed History")
    
    min_date = st.session_state.detection_history['timestamp'].min().date()
    max_date = st.session_state.detection_history['timestamp'].max().date()
    date_range = st.date_input(
        "Select date range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        filtered_history = st.session_state.detection_history[
            (st.session_state.detection_history['timestamp'].dt.date >= date_range[0]) &
            (st.session_state.detection_history['timestamp'].dt.date <= date_range[1])
        ]
    else:
        filtered_history = st.session_state.detection_history
    
    st.dataframe(
        filtered_history.sort_values('timestamp', ascending=False),
        column_config={
            "timestamp": "Timestamp",
            "cheating": "Cheating Incidents",
            "good": "Good Behavior"
        },
        use_container_width=True
    )
    
    if st.button("Export History to CSV"):
        csv = filtered_history.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="monitoring_history.csv",
            mime="text/csv"
        )

def system_settings():
    st.title("\U0001F527 System Configuration")
    
    st.subheader("Video Processing Settings")
    col1, col2 = st.columns(2)
    with col1:
        new_width = st.number_input("Frame Width", 320, 1920, st.session_state.frame_size[0])
    with col2:
        new_height = st.number_input("Frame Height", 240, 1080, st.session_state.frame_size[1])
    
    if st.button("Apply Resolution Settings"):
        st.session_state.frame_size = (new_width, new_height)
        st.success(f"Frame size set to {new_width}x{new_height}")
    
    st.subheader("Alert Settings")
    st.session_state.email_alerts = st.checkbox(
        "Enable Email Alerts",
        value=st.session_state.email_alerts,
        help="Send email notifications when cheating is detected"
    )
    
    st.subheader("Model Information")
    st.write(f"Current model: {model_path}")
    st.write(f"Model classes: {class_names}")
    
    st.subheader("System Diagnostics")
    if st.button("Run System Check"):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                st.success("Webcam access: Working")
                cap.release()
            else:
                st.error("Webcam access: Failed")
            
            test_model = YOLO(model_path)
            st.success("Model loading: Working")
            
            os.makedirs("cheating_detections", exist_ok=True)
            st.success("File system access: Working")
            
        except Exception as e:
            st.error(f"System check failed: {str(e)}")

# Main App
def main():
    st.set_page_config(
        page_title="ExamVisio Pro", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Request notification permissions
    st.components.v1.html("""
    <script>
        if (Notification.permission !== "granted" && Notification.permission !== "denied") {
            Notification.requestPermission();
        }
    </script>
    """, height=0)
    
    # --- AUTHENTICATION SECTION (directly in main) ---
    if not st.session_state.get('authentication_status'):
        st.title("Welcome to ExamVisio Pro")
        st.write("Please log in to access the admin panel")
        try:
            with open('config.yaml') as file:
                config = yaml.load(file, Loader=SafeLoader)
            
            # Create authenticator with unique key
            authenticator = stauth.Authenticate(
                config['credentials'],
                config['cookie']['name'],
                config['cookie']['key'],
                config['cookie']['expiry_days'],
            )
            
            # Login widget 
            # Login widget (with None check)
            login_result = authenticator.login(
            fields={'form_name': 'Login', 'username': 'Username', 'password': 'Password'},
            location='main')

            if login_result is None:
                st.warning("Please enter your username and password")
                return

            name, authentication_status, username = authenticator.login(
                fields={'form_name': 'Login', 'username': 'Username', 'password': 'Password'},
                location='main'
            )
            
            if authentication_status:
                st.session_state.authentication_status = True
                st.session_state.name = name
                st.session_state.username = username
                st.session_state.authenticator = authenticator
                st.experimental_rerun()
            elif authentication_status is False:
                st.error("Username/password is incorrect")
            elif authentication_status is None:
                st.warning("Please enter your username and password")
                
            # Registration section with unique key
            with st.expander("Don't have an account? Register here"):
                try:
                    'email_of_registered_user', \
                    'username_of_registered_user', \
                     authenticator.register_user(
                        location = 'main', 
                        preauthorization=False,
                        fields={
                            'form_name': 'Register user',
                            'Email': 'Email',
                            'Username': 'Username',
                            'password': 'Password',
                            'repeat_password': 'Repeat Password',
                            'register': 'Register'
                        }
                     )
                    if 'email_of_registered_user' in locals() and 'username_of_registered_user' in locals():
                      st.success("User registered successfully!")
                except Exception as e:
                    st.error(f"Registration error: {str(e)}")      
                    
        except FileNotFoundError:
            st.warning("Running in demo mode - no authentication config found")
            if st.button("Continue in Demo Mode", key='demo_btn'):
                st.session_state.authentication_status = True
                st.session_state.name = "Demo User"
                st.experimental_rerun()
                
        with st.expander("Reset password?"):
            try:
                if st.session_state.authenticator.get('authentication_status'):
                    if st.session_state.authenticator.reset_password(st.session_state.get('username'), 
                        fields={'form_name': 'Reset Password',
                                 'current_password': 'Current Password', 
                                 'new_password': 'New Password',
                                 'repeat_password': 'Repeat Password',
                                 'reset': 'Reset'}
                    ):
                     st.success("Password reset successful!")
            except Exception as e:
                st.error(f"Password reset error: {str(e)}")  

        with st.expander("Forgot your password?"): 
            try:
                if st.session_state.authenticator.get('authentication_status'):
                    if st.session_state.authenticator.forgot_password(
                          fields={'form_name': 'Forgot Password',
                                 'Email': 'Email',
                                 'forgot': 'Forgot Password'},
                        send_email=True,
                    ):
                        st.success("Password reset link sent to your email!")
            except Exception as e:
                st.error(f"Forgot password error: {str(e)}")            
        return
    # --- MAIN APPLICATION (only shown when authenticated) ---
    st.sidebar.title("Exam Visio Pro Admin Panel")
    
    # Logout button
    if st.sidebar.button("Logout"):
        if 'authenticator' in st.session_state and st.session_state.authenticator:
            try:
                st.session_state.authenticator.logout(fields={'form_name': 'Logout'}, location='main')
            except:
                pass
        
    
    st.sidebar.markdown(f"### 👤 Welcome, {st.session_state.name or 'Admin'}")
    
    pages = [
        "Live Monitoring",
        "Analytics Dashboard",
        "Evidence Review",
        "Alerts",
        "History",
        "System Settings"
    ]
    st.session_state.current_page = st.sidebar.radio("Navigate to:", pages)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Detection Settings")
    face_detection = st.sidebar.checkbox("Face Detection", value=True)
    noise_detection = st.sidebar.checkbox("Noise Detection", value=True)
    multi_face_detection = st.sidebar.checkbox("Multi-face Detection", value=True)
    confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    
    st.sidebar.markdown("---")
    global cheating_placeholder, non_cheating_placeholder, alert_placeholder, person_count_placeholder
    cheating_placeholder = st.sidebar.empty()
    non_cheating_placeholder = st.sidebar.empty()
    alert_placeholder = st.sidebar.empty()
    person_count_placeholder = st.sidebar.empty()
    
    # Page routing
    if st.session_state.current_page == "Live Monitoring":
        live_monitoring()
    elif st.session_state.current_page == "Analytics Dashboard":
        visualize_results()
    elif st.session_state.current_page == "Evidence Review":
        display_captured_images()
    elif st.session_state.current_page == "Alerts":
        alerts()
    elif st.session_state.current_page == "History":
        history_page()
    elif st.session_state.current_page == "System Settings":
        system_settings()

if __name__ == "__main__":
    main()