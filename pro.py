import os
import cv2
import time
import queue
import threading
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from PIL import Image

import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from oauthlib.oauth2 import WebApplicationClient
import requests
import smtplib
import json

from ultralytics import YOLO
import torch

# ========================
# CONFIGURATION MANAGEMENT
# ========================
class Config:
    """Centralized configuration management"""
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    ALERT_CONFIDENCE = 0.6
    MODEL_PATH = "runs/detect/train22/weights/best.pt"
    
    # Video processing
    FRAME_SIZE = (640, 480)
    TARGET_FPS = 10
    ALERT_COOLDOWN = 5  # seconds
    
    # Alerts
    EMAIL_ALERTS = False
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_ADDRESS = os.getenv("ALERT_EMAIL", "alerts@examvisio.com")
    EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
    RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "admin@examvisio.com")
    
    # Authentication
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
    
    # Paths
    EVIDENCE_DIR = "cheating_detections"
    
    @classmethod
    def initialize(cls):
        """Initialize configuration and required directories"""
        os.makedirs(cls.EVIDENCE_DIR, exist_ok=True)
        
        # Load class names
        cls.CLASS_NAMES = {0: "cheating", 1: "good"}
        cls.CLASS_COLORS = {"cheating": (0, 0, 255), "good": (0, 255, 0)}

# Initialize configuration
Config.initialize()

# ====================
# VIDEO PROCESSING
# ====================
class VideoProcessor:
    """Handles video processing from both webcam and file sources"""
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.capture_thread = None
        self.monitoring_active = False
        self.last_alert_time = 0
        self.temp_files = []
        self.current_frame = None
        self.latest_results = None
        self.latest_counts = {'cheating': 0, 'good': 0}

    def start_webcam_monitoring(self):
        """Start monitoring from webcam"""
        self.monitoring_active = True
        self.capture_thread = threading.Thread(
            target=self._webcam_capture_loop,
            daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self._continuous_processing_loop,
            daemon=True
        )
        self.capture_thread.start()
        self.processing_thread.start()

    def start_file_monitoring(self, file_path):
        """Start monitoring from video file"""
        self.monitoring_active = True
        self.temp_files.append(file_path)  # Track for cleanup
        self.capture_thread = threading.Thread(
            target=self._file_capture_loop,
            args=(file_path,),
            daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self._continuous_processing_loop,
            daemon=True
        )
        self.capture_thread.start()
        self.processing_thread.start()

    def stop_monitoring(self):
        """Stop all monitoring activities"""
        self.monitoring_active = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        self._cleanup_temp_files()
        self.frame_queue = queue.Queue(maxsize=10)
        self.current_frame = None
        self.latest_results = None

    def _webcam_capture_loop(self):
        """Webcam capture thread"""
        cap = cv2.VideoCapture(0)
        try:
            while self.monitoring_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, Config.FRAME_SIZE)
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
        finally:
            cap.release()

    def _file_capture_loop(self, file_path):
        """Video file capture thread"""
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1 / fps if fps > 0 else 0.03
        
        try:
            while self.monitoring_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, Config.FRAME_SIZE)
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                time.sleep(frame_delay)
        finally:
            cap.release()

    def _continuous_processing_loop(self):
        """Continuous frame processing that runs independently of UI"""
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        
        while self.monitoring_active:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.current_frame = frame.copy()
                    
                    # Process frame for detections
                    results = self.process_frame(frame)
                    self.latest_results = results
                    
                    # Update counts using session state data manager
                    if 'data_manager' in st.session_state:
                        counts = st.session_state.data_manager.update_counts(results)
                        self.latest_counts = counts
                        
                        # Check for alerts
                        cheating_boxes = [
                            box for result in results for box in result.boxes
                            if Config.CLASS_NAMES.get(int(box.cls), '').lower() == 'cheating'
                            and float(box.conf) >= Config.ALERT_CONFIDENCE
                        ]
                        
                        if cheating_boxes and self.should_trigger_alert():
                            people_count = len(cheating_boxes)
                            alert_data = AlertManager.create_alert(frame, results, people_count)
                            
                            # Add to session state alert history
                            if 'alert_history' not in st.session_state:
                                st.session_state.alert_history = []
                            st.session_state.alert_history.append(alert_data)
                            
                            # Send email alert if enabled
                            if st.session_state.get('email_alerts', False):
                                threading.Thread(
                                    target=AlertManager.send_email_alert,
                                    args=(alert_data,),
                                    daemon=True
                                ).start()
                
                time.sleep(0.1)  # Small delay to prevent high CPU usage
                
            except Exception as e:
                print(f"Error in continuous processing: {e}")
                time.sleep(1)  # Wait before retrying
    
    def get_latest_frame_and_results(self):
        """Get the latest processed frame and results"""
        if self.current_frame is not None and self.latest_results is not None:
            # Return annotated frame
            annotated_frame = AlertManager._annotate_frame(
                self.current_frame.copy(), 
                self.latest_results
            )
            return annotated_frame, self.latest_results, self.latest_counts
        return None, None, {'cheating': 0, 'good': 0}

    def process_frame(self, frame):
        """Process a single frame for detections"""
        results = self.model.predict(
            frame,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            imgsz=640,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        return results

    def should_trigger_alert(self):
        """Check if we should trigger a new alert (cooldown)"""
        current_time = time.time()
        if (current_time - self.last_alert_time) > Config.ALERT_COOLDOWN:
            self.last_alert_time = current_time
            return True
        return False

    def _cleanup_temp_files(self):
        """Clean up temporary video files"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error cleaning up temp file {file_path}: {str(e)}")
        self.temp_files = []

# ====================
# ALERT MANAGEMENT
# ====================
class AlertManager:
    """Handles all alert-related functionality"""
    @staticmethod
    def create_alert(frame, results, people_count):
        """Create and store an alert"""
        max_confidence = max((float(box.conf) for result in results for box in result.boxes), default=0)
        timestamp = datetime.now()
        
        # Save alert image
        alert_image = AlertManager._annotate_frame(frame.copy(), results)
        image_path = os.path.join(
            Config.EVIDENCE_DIR,
            f"cheating_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{max_confidence:.2f}.jpg"
        )
        cv2.imwrite(image_path, alert_image)
        
        # Create alert data
        alert_data = {
            'timestamp': timestamp,
            'image_path': image_path,
            'confidence': max_confidence,
            'people_count': people_count
        }
        
        return alert_data

    @staticmethod
    def _annotate_frame(frame, results):
        """Annotate frame with detection information"""
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = Config.CLASS_NAMES.get(class_id, 'Unknown')
                confidence = float(box.conf)
                color = Config.CLASS_COLORS.get(class_name.lower(), (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f'{class_name} {confidence:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color, -1
                )
                cv2.putText(
                    frame, label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 255, 255), 1
                )
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(
            frame, timestamp,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )
        
        return frame

    @staticmethod
    def send_email_alert(alert_data):
        """Send email alert with detection details"""
        if not Config.EMAIL_ALERTS:
            return False

        msg = MIMEMultipart()
        msg['From'] = Config.EMAIL_ADDRESS
        msg['To'] = Config.RECIPIENT_EMAIL
        msg['Subject'] = f"ExamVisio Alert - Cheating Detected ({alert_data['confidence']:.2f})"
        
        body = f"""
        <h2>Cheating Incident Detected</h2>
        <p><strong>Time:</strong> {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Confidence:</strong> {alert_data['confidence']:.2f}</p>
        <p><strong>People in frame:</strong> {alert_data['people_count']}</p>
        <p>See attached image for details.</p>
        """
        msg.attach(MIMEText(body, 'html'))
        
        try:
            with open(alert_data['image_path'], 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', 
                             filename=os.path.basename(alert_data['image_path']))
                msg.attach(img)
            
            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
            server.starttls()
            server.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

# ====================
# AUTHENTICATION
# ====================
class AuthManager:
    """Handles user authentication and session management"""
    def __init__(self):
        self.authenticator = None
        self.auth_config = None
        self.load_auth_config()
        
    def load_auth_config(self):
        """Load authentication configuration from file"""
        try:
            with open('auth_config.yaml') as file:
                self.auth_config = yaml.load(file, Loader=SafeLoader)
                self.authenticator = stauth.Authenticate(
                    self.auth_config['credentials'],
                    self.auth_config['cookie']['name'],
                    self.auth_config['cookie']['key'],
                    self.auth_config['cookie']['expiry_days'],
                )
        except FileNotFoundError:
            self.auth_config = None
            self.authenticator = None

    def authenticate_user(self):
        """Handle user authentication"""
        if not self.authenticator:
            return False
            
        name, authentication_status, username = self.authenticator.login(
            'Login', 'main')
        
        if authentication_status:
            st.session_state.update({
                'name': name,
                'username': username,
                'authentication_status': True
            })
            return True
        elif authentication_status is False:
            st.error("Username/password is incorrect")
        elif authentication_status is None:
            st.warning("Please enter your username and password")
            
        return False

    def show_auth_ui(self):
        """Display authentication UI components"""
        if not self.authenticator:
            st.warning("Running in demo mode - no authentication config found")
            if st.button("Continue in Demo Mode"):
                st.session_state.authentication_status = True
                st.session_state.name = "Demo User"
                st.rerun()
            return False
            
        # Login form
        if not st.session_state.get('authentication_status'):
            if self.authenticate_user():
                return True
                
            # Registration
            with st.expander("Don't have an account? Register here"):
                try:
                    if self.authenticator.register_user(
                        'Register user', 'main', preauthorization=False
                    ):
                        st.success("User registered successfully!")
                except Exception as e:
                    st.error(f"Registration error: {str(e)}")
                    
            # Password reset
            with st.expander("Reset password"):
                try:
                    if self.authenticator.reset_password(
                        st.session_state.get('username'), 'Reset password', 'main'
                    ):
                        st.success("Password modified successfully")
                except Exception as e:
                    st.error(f"Password reset error: {str(e)}")
                    
            return False
        return True

# ====================
# DATA MANAGEMENT
# ====================
class DataManager:
    """Manages detection data and history"""
    def __init__(self):
        if 'detection_counts' not in st.session_state:
            st.session_state.detection_counts = {'cheating': 0, 'good': 0}
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = pd.DataFrame(
                columns=['timestamp', 'cheating', 'good'])
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []

    def update_counts(self, results):
        """Update detection counts based on model results"""
        counts = {'cheating': 0, 'good': 0}
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = Config.CLASS_NAMES.get(class_id, 'Unknown')
                if class_name.lower() == 'cheating':
                    counts['cheating'] += 1
                else:
                    counts['good'] += 1
        
        # Update session state
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

    def get_filtered_alerts(self, min_confidence=0.5, min_people=1):
        """Get alerts filtered by criteria"""
        return [
            alert for alert in st.session_state.alert_history
            if alert['confidence'] >= min_confidence 
            and alert['people_count'] >= min_people
        ]

    def export_history(self, format='csv'):
        """Export detection history"""
        if format == 'csv':
            return st.session_state.detection_history.to_csv(index=False)
        elif format == 'json':
            return st.session_state.detection_history.to_json(orient='records')
        return None

# ====================
# STREAMLIT UI
# ====================
class ExamVisioUI:
    """Main application UI and page management"""
    def __init__(self):
        # Use session state for persistent video processor
        if 'video_processor' not in st.session_state:
            st.session_state.video_processor = VideoProcessor()
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager()
            
        self.video_processor = st.session_state.video_processor
        self.data_manager = st.session_state.data_manager
        self.auth_manager = AuthManager()
        
        # Initialize session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Live Monitoring"
        if 'email_alerts' not in st.session_state:
            st.session_state.email_alerts = Config.EMAIL_ALERTS
        
        # Configure Streamlit
        st.set_page_config(
            page_title="ExamVisio Pro", 
            layout="wide", 
            initial_sidebar_state="expanded"
        )
        
        # Request notification permissions
        self._request_notification_permission()

    def _request_notification_permission(self):
        """Request browser notification permissions"""
        st.components.v1.html("""
        <script>
            if (Notification.permission !== "granted" && Notification.permission !== "denied") {
                Notification.requestPermission();
            }
        </script>
        """, height=0)

    def run(self):
        """Main application loop"""
        if not st.session_state.get('authentication_status'):
            if not self.auth_manager.show_auth_ui():
                return
                
        self._render_sidebar()
        self._render_current_page()

    def _render_sidebar(self):
        """Render the application sidebar"""
        st.sidebar.title("ExamVisio Pro")
        st.sidebar.markdown(f"### 👤 Welcome, {st.session_state.get('name', 'Admin')}")
        
        # Show monitoring status at top of sidebar
        if self.video_processor.monitoring_active:
            st.sidebar.success("🔴 LIVE - Monitoring Active")
        else:
            st.sidebar.info("⚪ Monitoring Stopped")
        
        # Navigation
        pages = [
            "Live Monitoring",
            "Analytics Dashboard",
            "Evidence Review",
            "Alerts",
            "History",
            "System Settings"
        ]
        st.session_state.current_page = st.sidebar.radio("📌 Navigation", pages)
        
        # Detection settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Detection Settings")
        Config.CONFIDENCE_THRESHOLD = st.sidebar.slider(
            "Confidence Threshold", 0.1, 0.9, Config.CONFIDENCE_THRESHOLD, 0.05)
        Config.ALERT_CONFIDENCE = st.sidebar.slider(
            "Alert Confidence", 0.1, 0.9, Config.ALERT_CONFIDENCE, 0.05)
        st.session_state.email_alerts = st.sidebar.checkbox(
            "Email Alerts", value=st.session_state.email_alerts)
        
        # Live metrics - always update regardless of current page
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Live Metrics")
        
        # Get current metrics from video processor
        if self.video_processor.monitoring_active:
            _, _, current_counts = self.video_processor.get_latest_frame_and_results()
            people_count = current_counts['cheating'] + current_counts['good']
        else:
            current_counts = {'cheating': 0, 'good': 0}
            people_count = 0
            
        # Update metrics display
        st.sidebar.metric(
            "Cheating Detections", 
            st.session_state.detection_counts['cheating'])
        st.sidebar.metric(
            "Good Behavior", 
            st.session_state.detection_counts['good'])
        st.sidebar.metric(
            "Active Alerts", 
            len(st.session_state.alert_history))
        st.sidebar.metric(
            "People in Frame", 
            people_count)
        
        # Logout button
        if st.sidebar.button("Logout"):
            # Stop monitoring before logout
            if self.video_processor.monitoring_active:
                self.video_processor.stop_monitoring()
            st.session_state.clear()
            st.experimental_rerun()

    def _render_current_page(self):
        """Render the current active page"""
        if st.session_state.current_page == "Live Monitoring":
            self._render_live_monitoring()
        elif st.session_state.current_page == "Analytics Dashboard":
            self._render_analytics()
        elif st.session_state.current_page == "Evidence Review":
            self._render_evidence()
        elif st.session_state.current_page == "Alerts":
            self._render_alerts()
        elif st.session_state.current_page == "History":
            self._render_history()
        elif st.session_state.current_page == "System Settings":
            self._render_settings()

    def _render_live_monitoring(self):
        """Render the live monitoring page"""
        st.title("📹 Live Monitoring Dashboard")
        
        # Show monitoring status
        if self.video_processor.monitoring_active:
            st.success("🔴 LIVE - Monitoring is active and persistent across all pages")
        else:
            st.info("⚪ No monitoring active")
        
        source = st.selectbox("Select Video Source", ("Webcam", "Video File"))
        
        # Video display
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if source == "Webcam":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Monitoring", disabled=self.video_processor.monitoring_active):
                    self.video_processor.start_webcam_monitoring()
                    st.success("✅ Webcam monitoring started!")
                    time.sleep(1)
                    st.rerun()
            with col2:
                if st.button("Stop Monitoring", disabled=not self.video_processor.monitoring_active):
                    self.video_processor.stop_monitoring()
                    st.warning("⏹️ Monitoring stopped")
                    time.sleep(1)
                    st.rerun()
                    
        elif source == "Video File":
            uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                    tmpfile.write(uploaded_file.read())
                    tmpfile_path = tmpfile.name
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Start Processing", disabled=self.video_processor.monitoring_active):
                        self.video_processor.start_file_monitoring(tmpfile_path)
                        st.success("✅ File processing started!")
                        time.sleep(1)
                        st.rerun()
                with col2:
                    if st.button("Stop Processing", disabled=not self.video_processor.monitoring_active):
                        self.video_processor.stop_monitoring()
                        st.warning("⏹️ Processing stopped")
                        time.sleep(1)
                        st.rerun()
        
        # Display current frame if monitoring is active
        if self.video_processor.monitoring_active:
            frame, results, counts = self.video_processor.get_latest_frame_and_results()
            if frame is not None:
                video_placeholder.image(frame, channels='BGR', caption="Live Feed")
                people_count = counts['cheating'] + counts['good']
                status_placeholder.success(f"🟢 Active - People in frame: {people_count}")
            else:
                video_placeholder.info("📷 Waiting for camera frame...")
                status_placeholder.info("Initializing...")
        else:
            video_placeholder.info("📷 Start monitoring to see live feed")
            status_placeholder.info("Monitoring not active")
            
        # Manual refresh button
        if st.button("🔄 Refresh Display"):
            st.rerun()

    def _render_analytics(self):
        """Render the analytics dashboard"""
        st.title("📊 Analytics Dashboard")
        
        # Show monitoring status on non-monitoring pages
        if self.video_processor.monitoring_active:
            st.info("🔴 Live monitoring is running in background - Detection continues while viewing analytics")
        
        if st.session_state.detection_history.empty:
            st.info("No data available. Start monitoring to collect analytics.")
            return
            
        st.subheader("Detection Overview")
        
        # Time window filter
        time_window = st.selectbox(
            "Time Window", 
            ["Last 5 minutes", "Last 15 minutes", "Last hour", "All time"])
        
        # Filter data
        now = datetime.now()
        if time_window == "Last 5 minutes":
            filtered = st.session_state.detection_history[
                st.session_state.detection_history['timestamp'] > now - pd.Timedelta(minutes=5)]
        elif time_window == "Last 15 minutes":
            filtered = st.session_state.detection_history[
                st.session_state.detection_history['timestamp'] > now - pd.Timedelta(minutes=15)]
        elif time_window == "Last hour":
            filtered = st.session_state.detection_history[
                st.session_state.detection_history['timestamp'] > now - pd.Timedelta(hours=1)]
        else:
            filtered = st.session_state.detection_history
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["Counts", "Timeline", "Proportions"])
        
        with tab1:
            fig, ax = plt.subplots()
            sns.barplot(
                x=['Cheating', 'Good'], 
                y=[st.session_state.detection_counts['cheating'], 
                   st.session_state.detection_counts['good']],
                palette=['red', 'green'],
                ax=ax
            )
            ax.set_title('Behavior Detection Counts')
            st.pyplot(fig)
            
        with tab2:
            if not filtered.empty:
                resampled = filtered.set_index('timestamp').resample('30S').sum()
                fig, ax = plt.subplots(figsize=(10, 4))
                resampled.plot(ax=ax)
                ax.set_title('Detection Timeline')
                st.pyplot(fig)
            else:
                st.warning("No data for selected time window")
                
        with tab3:
            fig, ax = plt.subplots()
            ax.pie(
                [st.session_state.detection_counts['cheating'], 
                 st.session_state.detection_counts['good']],
                labels=['Cheating', 'Good'], 
                autopct='%1.1f%%',
                colors=['red', 'green']
            )
            ax.set_title('Behavior Proportions')
            st.pyplot(fig)

    def _render_evidence(self):
        """Render the evidence review page"""
        st.title("🖼️ Evidence Gallery")
        
        # Show monitoring status on non-monitoring pages
        if self.video_processor.monitoring_active:
            st.info("🔴 Live monitoring is running in background - New evidence may appear")
        
        if not st.session_state.alert_history:
            st.info("No cheating incidents detected yet.")
            return
            
        st.subheader("Detected Incidents")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_confidence = st.slider(
                "Minimum Confidence", 0.0, 1.0, 0.6, 0.05)
        with col2:
            min_people = st.slider(
                "Minimum People", 1, 10, 1)
        
        # Filter alerts
        filtered = [
            alert for alert in st.session_state.alert_history
            if alert['confidence'] >= min_confidence 
            and alert['people_count'] >= min_people
        ]
        
        if not filtered:
            st.warning("No alerts match your criteria")
        else:
            for alert in reversed(filtered):
                with st.expander(
                    f"Alert at {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"Confidence: {alert['confidence']:.2f}"):
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(alert['image_path'])
                    with col2:
                        st.write(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Confidence:** {alert['confidence']:.2f}")
                        st.write(f"**People:** {alert['people_count']}")
                        
                        with open(alert['image_path'], 'rb') as f:
                            st.download_button(
                                "Download Evidence",
                                f.read(),
                                file_name=os.path.basename(alert['image_path'])
                            )

    def _render_alerts(self):
        """Render the alerts management page"""
        st.title("🚨 Alert Management")
        
        if not st.session_state.alert_history:
            st.success("No alerts generated!")
            return
            
        st.warning(f"**{len(st.session_state.alert_history)} cheating incidents detected**")
        
        # Filters
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        with col1:
            severity = st.select_slider(
                "Severity", 
                options=['Low', 'Medium', 'High'], 
                value='Medium')
        with col2:
            date_filter = st.date_input("Filter by date")
        
        # Determine confidence threshold based on severity
        if severity == 'Low':
            min_conf = 0.5
        elif severity == 'Medium':
            min_conf = 0.7
        else:
            min_conf = 0.85
            
        # Filter alerts
        filtered = [
            alert for alert in st.session_state.alert_history
            if alert['confidence'] >= min_conf and
            (not date_filter or alert['timestamp'].date() == date_filter)
        ]
        
        if not filtered:
            st.warning("No alerts match your filters")
        else:
            for i, alert in enumerate(reversed(filtered), 1):
                with st.container():
                    cols = st.columns([1, 3, 1])
                    with cols[0]:
                        st.image(alert['image_path'], width=200)
                    with cols[1]:
                        st.write(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Confidence:** {alert['confidence']:.2f}")
                        st.write(f"**People:** {alert['people_count']}")
                    with cols[2]:
                        with open(alert['image_path'], 'rb') as f:
                            st.download_button(
                                "Download",
                                f.read(),
                                file_name=f"evidence_{i}.jpg",
                                key=f"dl_{i}"
                            )
                    st.markdown("---")

    def _render_history(self):
        """Render the monitoring history page"""
        st.title("📜 Monitoring History")
        
        if st.session_state.detection_history.empty:
            st.info("No monitoring history available yet.")
            return
            
        st.subheader("Session Summary")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Total Sessions",
            len(st.session_state.detection_history['timestamp'].dt.date.unique()))
        col2.metric(
            "Total Alerts",
            len(st.session_state.alert_history))
        col3.metric(
            "Avg Cheating per Session",
            f"{st.session_state.detection_history['cheating'].mean():.1f}")
        
        # Date range filter
        st.subheader("Detailed History")
        min_date = st.session_state.detection_history['timestamp'].min().date()
        max_date = st.session_state.detection_history['timestamp'].max().date()
        date_range = st.date_input(
            "Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data
        if len(date_range) == 2:
            filtered = st.session_state.detection_history[
                (st.session_state.detection_history['timestamp'].dt.date >= date_range[0]) &
                (st.session_state.detection_history['timestamp'].dt.date <= date_range[1])]
        else:
            filtered = st.session_state.detection_history
            
        # Display data
        st.dataframe(
            filtered.sort_values('timestamp', ascending=False),
            column_config={
                "timestamp": "Timestamp",
                "cheating": "Cheating",
                "good": "Good Behavior"
            },
            use_container_width=True
        )
        
        # Export button
        if st.button("Export to CSV"):
            csv = self.data_manager.export_history('csv')
            st.download_button(
                "Download CSV",
                csv,
                "monitoring_history.csv",
                "text/csv"
            )

    def _render_settings(self):
        """Render the system settings page"""
        st.title("⚙️ System Settings")
        
        st.subheader("Video Processing")
        col1, col2 = st.columns(2)
        with col1:
            new_width = st.number_input(
                "Frame Width", 
                320, 1920, Config.FRAME_SIZE[0])
        with col2:
            new_height = st.number_input(
                "Frame Height", 
                240, 1080, Config.FRAME_SIZE[1])
                
        if st.button("Apply Resolution"):
            Config.FRAME_SIZE = (new_width, new_height)
            st.success("Resolution updated")
            
        st.subheader("Alert Settings")
        Config.EMAIL_ALERTS = st.checkbox(
            "Enable Email Alerts",
            value=Config.EMAIL_ALERTS)
        Config.ALERT_COOLDOWN = st.number_input(
            "Alert Cooldown (seconds)",
            1, 60, Config.ALERT_COOLDOWN)
            
        st.subheader("Model Information")
        st.write(f"Model path: {Config.MODEL_PATH}")
        st.write(f"Classes: {Config.CLASS_NAMES}")
        
        st.subheader("System Diagnostics")
        if st.button("Run Diagnostics"):
            with st.spinner("Running diagnostics..."):
                self._run_diagnostics()

    def _run_diagnostics(self):
        """Run system diagnostics"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hardware**")
            st.write(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.warning("No GPU detected - using CPU")
                
        with col2:
            st.markdown("**Software**")
            st.write(f"OpenCV Version: {cv2.__version__}")
            st.write(f"PyTorch Version: {torch.__version__}")
            st.write(f"Streamlit Version: {st.__version__}")
            
        st.markdown("**System Checks**")
        
        # Webcam check
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                st.success("Webcam: Working")
                cap.release()
            else:
                st.error("Webcam: Not accessible")
        except:
            st.error("Webcam: Error during check")
            
        # Model check
        try:
            test_model = YOLO(Config.MODEL_PATH)
            st.success("Model: Loaded successfully")
        except:
            st.error("Model: Failed to load")
            
        # Filesystem check
        try:
            os.makedirs(Config.EVIDENCE_DIR, exist_ok=True)
            test_file = os.path.join(Config.EVIDENCE_DIR, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.unlink(test_file)
            st.success("Filesystem: Read/Write OK")
        except:
            st.error("Filesystem: Permission issues")

# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    app = ExamVisioUI()
    app.run()