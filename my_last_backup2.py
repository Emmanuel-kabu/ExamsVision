import os
import cv2
import time
import queue
import threading
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
from contextlib import contextmanager
import matplotlib.pyplot as plt
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from PIL import Image
import logging
import smtplib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from ultralytics import YOLO
import torch
from AUTH import sign_in_with_email, sign_up_with_email, sign_out, forget_password, auth_screen

# ========================
# CONFIGURATION MANAGEMENT
# ========================
class Config:
    """Centralized configuration management"""
    # Model parameters
    MODEL_PATH = "runs/detect/train22/weights/best.pt"
    CLASS_NAMES = {0: "cheating", 1: "good"}
    CLASS_COLORS = {"cheating": (0, 0, 255), "good": (0, 255, 0)}
    
    # Video processing
    FRAME_SIZE = (640, 480)
    TARGET_FPS = 30
    ALERT_COOLDOWN = 5  # seconds
    
    # Alerts
    EMAIL_ALERTS = False
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_ADDRESS = os.getenv("ALERT_EMAIL", "alerts@examvisio.com")
    EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
    RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "admin@examvisio.com")
    
    # Paths
    EVIDENCE_DIR = "cheating_detections"
    
    @classmethod
    def initialize(cls):
        """Initialize configuration"""
        os.makedirs(cls.EVIDENCE_DIR, exist_ok=True)

# Initialize configuration
Config.initialize()

# ====================
# VIDEO PROCESSING
# ====================
class VideoProcessor:
    """Thread-safe video processing with camera management"""
    def __init__(self):
        # Threading controls
        self._monitoring_event = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._lock = threading.Lock()
        
        # Video resources
        self.frame_queue = queue.Queue(maxsize=3)  # Small buffer
        self.last_alert_time = 0
        self._last_error = None

    def start_monitoring(self, video_source=0) -> bool:
        """Start video monitoring thread-safe"""
        with self._lock:
            if self._monitoring_event.is_set():
                logger.warning("Monitoring already active")
                return False

            self._stop_event.clear()
            self._last_error = None
            
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(video_source,),
                daemon=True
            )
            self._capture_thread.start()
            
            # Wait for thread to initialize
            if not self._monitoring_event.wait(2.0):
                self._stop_event.set()
                logger.error("Capture thread failed to start")
                return False
            
            return True

    def _capture_loop(self, video_source):
        """Main capture loop with error recovery"""
        self._monitoring_event.set()
        retry_count = 0
        max_retries = 3
        
        try:
            while not self._stop_event.is_set() and retry_count < max_retries:
                try:
                    with self._open_capture(video_source) as cap:
                        logger.info("Capture started successfully")
                        retry_count = 0  # Reset on success
                        
                        while not self._stop_event.is_set():
                            ret, frame = cap.read()
                            if not ret:
                                logger.warning("Frame read failed")
                                time.sleep(0.1)
                                continue
                            
                            # Process and queue frame
                            self._process_frame(frame)
                            
                except Exception as e:
                    retry_count += 1
                    self._last_error = str(e)
                    logger.error(f"Capture error (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(1.0)
                    
        finally:
            self._monitoring_event.clear()
            logger.info("Capture thread stopped")

    def _process_frame(self, frame):
        """Process and queue a frame"""
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()  # Discard oldest if full
            self.frame_queue.put_nowait(frame)
        except queue.Empty:
            pass

    @contextmanager
    def _open_capture(self, source):
        """Context manager for video capture"""
        cap = None
        try:
            # Try multiple backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(source, backend)
                    if cap.isOpened():
                        # Configure camera
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_SIZE[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_SIZE[1])
                        cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                        
                        # Warmup
                        for _ in range(5):
                            cap.read()
                            time.sleep(0.1)
                        
                        yield cap
                        return
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    if cap:
                        cap.release()
            
            raise RuntimeError("All capture backends failed")
        finally:
            if cap:
                cap.release()

    def stop_monitoring(self) -> bool:
        """Stop monitoring safely"""
        with self._lock:
            if not self._monitoring_event.is_set():
                return True
                
            self._stop_event.set()
            
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=1.0)
                return not self._capture_thread.is_alive()
                
            return True

    def is_monitoring(self) -> bool:
        """Check if monitoring is active"""
        return self._monitoring_event.is_set()

    def get_latest_frame(self):
        """Get the latest frame non-blocking"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def should_trigger_alert(self):
        """Check alert cooldown"""
        return (time.time() - self.last_alert_time) > Config.ALERT_COOLDOWN

# ====================
# ALERT MANAGEMENT
# ====================
class AlertManager:
    """Handles alert creation and notifications"""
    @staticmethod
    def create_alert(frame, detections):
        """Create alert evidence"""
        timestamp = datetime.now()
        max_confidence = max((float(box.conf) for det in detections for box in det.boxes), default=0)
        
        # Annotate frame
        annotated_frame = AlertManager._annotate_frame(frame.copy(), detections)
        
        # Save evidence
        filename = f"alert_{timestamp.strftime('%Y%m%d_%H%M%S')}_{max_confidence:.2f}.jpg"
        image_path = os.path.join(Config.EVIDENCE_DIR, filename)
        cv2.imwrite(image_path, annotated_frame)
        
        return {
            'timestamp': timestamp,
            'image_path': image_path,
            'confidence': max_confidence,
            'people_count': sum(len(det.boxes) for det in detections)
        }

    @staticmethod
    def _annotate_frame(frame, results):
        """Add detection annotations to frame"""
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = Config.CLASS_NAMES.get(class_id, 'Unknown')
                confidence = float(box.conf)
                color = Config.CLASS_COLORS.get(class_name.lower(), (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add timestamp
        cv2.putText(frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                   (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    @staticmethod
    def send_email_alert(alert_data):
        """Send email notification"""
        if not Config.EMAIL_ALERTS:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_ADDRESS
            msg['To'] = Config.RECIPIENT_EMAIL
            msg['Subject'] = f"Cheating Alert ({alert_data['confidence']:.2f})"
            
            body = f"""
            <h2>Cheating Detected</h2>
            <p><b>Time:</b> {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><b>Confidence:</b> {alert_data['confidence']:.2f}</p>
            <p><b>People:</b> {alert_data['people_count']}</p>
            """
            msg.attach(MIMEText(body, 'html'))
            
            with open(alert_data['image_path'], 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', 
                             filename=os.path.basename(alert_data['image_path']))
                msg.attach(img)
            
            with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
                server.starttls()
                server.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
                server.send_message(msg)
            
            return True
        except Exception as e:
            logger.error(f"Email failed: {e}")
            return False

# ====================
# DATA MANAGEMENT
# ====================
class DataManager:
    """Manages detection data and history"""
    def __init__(self):
        # Initialize session state
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = pd.DataFrame(
                columns=['timestamp', 'cheating', 'good'])
        
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
        
        if 'detection_counts' not in st.session_state:
            st.session_state.detection_counts = {'cheating': 0, 'good': 0}

    def update_detections(self, results):
        """Update detection counts and history"""
        counts = {'cheating': 0, 'good': 0}
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if Config.CLASS_NAMES.get(class_id) == 'cheating':
                    counts['cheating'] += 1
                else:
                    counts['good'] += 1
        
        # Update counts
        st.session_state.detection_counts['cheating'] += counts['cheating']
        st.session_state.detection_counts['good'] += counts['good']
        
        # Add to history
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

    def add_alert(self, alert_data):
        """Add new alert to history"""
        st.session_state.alert_history.append(alert_data)

    def get_filtered_alerts(self, min_confidence=0.5, min_people=1):
        """Get filtered alerts"""
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
# MODEL MANAGEMENT
# ====================
class ModelManager:
    """Handles YOLO model operations"""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

    def detect(self, frame):
        """Run detection on frame"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Resize frame to configured size
        frame = cv2.resize(frame, Config.FRAME_SIZE)
        
        # Run detection
        results = self.model(frame, conf=Config.CONFIDENCE_THRESHOLD)
        return results

# ====================
# STREAMLIT UI
# ====================
class ExamVisioUI:
    """Main application UI with all pages"""
    def __init__(self):
        # Initialize core components
        self.model_manager = ModelManager(Config.MODEL_PATH)
        self.data_manager = DataManager()
        self.alert_manager = AlertManager()
        
        # Initialize session state
        if 'video_processor' not in st.session_state:
            st.session_state.video_processor = VideoProcessor()
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Live Monitoring"
        
        if 'email_alerts' not in st.session_state:
            st.session_state.email_alerts = Config.EMAIL_ALERTS
            
        if 'authentication_status' not in st.session_state:
            st.session_state.authentication_status = False
            
        if 'name' not in st.session_state:
            st.session_state.name = None
        
        # Configure Streamlit
        st.set_page_config(
            page_title="ExamVisio Pro",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def run(self):
        """Run the application"""
        # Check if user is authenticated
        if not st.session_state.get('authentication_status'):
            self._render_auth_screen()
            return
            
        self._render_sidebar()
        self._render_current_page()

    def _render_sidebar(self):
        """Render the application sidebar"""
        st.sidebar.title("ExamVisio Pro")
        st.sidebar.markdown(f"### 👤 Welcome, {st.session_state.get('name', 'Admin')}")
        
        # Navigation
        pages = {
            "Live Monitoring": self._render_live_monitoring,
            "Analytics Dashboard": self._render_analytics,
            "Evidence Review": self._render_evidence,
            "Alerts": self._render_alerts,
            "History": self._render_history,
            "System Settings": self._render_settings
        }
        st.session_state.current_page = st.sidebar.radio("📌 Navigation", list(pages.keys()))
        
        # Detection settings
        # Detection settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Detection Settings")
        Config.CONFIDENCE_THRESHOLD = st.sidebar.slider(
            "Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
        Config.ALERT_CONFIDENCE = st.sidebar.slider(
            "Alert Threshold", 0.1, 0.9, 0.6, 0.05)
        st.session_state.email_alerts = st.sidebar.checkbox(
            "Email Alerts", value=st.session_state.email_alerts)

        # Metrics display
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Live Metrics")
        self.cheating_metric = st.sidebar.empty()
        self.good_metric = st.sidebar.empty()
        self.alerts_metric = st.sidebar.empty()
        self.people_metric = st.sidebar.empty()
        
        # Sign Out Button (at the bottom of sidebar)
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Sign Out"):
            sign_out()
            st.session_state.authentication_status = False
            st.session_state.name = None
            st.rerun()

    def _render_sidebar_metrics(self):
        """Render sidebar metrics"""
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Cheating", st.session_state.detection_counts['cheating'])
        col2.metric("Good", st.session_state.detection_counts['good'])

        st.sidebar.metric("Alerts", len(st.session_state.alert_history))

    def _render_current_page(self):
        """Render the current active page"""
        pages = {
            "Live Monitoring": self._render_live_monitoring,
            "Analytics Dashboard": self._render_analytics,
            "Evidence Review": self._render_evidence,
            "Alerts": self._render_alerts,
            "History": self._render_history,
            "System Settings": self._render_settings
        }
        
        pages[st.session_state.current_page]()

    def _render_live_monitoring(self):
        """Live monitoring page"""
        st.title("🎥 Live Monitoring")
        
        # Video source selection
        source_type = st.radio("Video Source", ["Webcam", "File"])
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Monitoring"):
                self._start_monitoring(source_type)
        with col2:
            if st.button("Stop Monitoring"):
                self._stop_monitoring()
        
        # Video display
        self._render_video_feed()

    def _render_sign_in(self):
        """Render sign-in form"""
        st.title("Sign In")
        email = st.text_input("Email", key="email_signin")
        password = st.text_input("Password", type="password", key="password_signin")
        
        if st.button("Sign In"):
            try:
                user = sign_in_with_email(email, password)
                if user:
                    st.session_state.current_page = "Live Monitoring"
                    st.rerun()
            except Exception as e:
                st.error(f"Sign in failed: {str(e)}")

    def _render_forgot_password(self):
           """Render forgot password form"""
           st.title("Forgot Password")
           email = st.text_input("Email", key="email_forgot")
           if st.button("Send Reset Link"):
              forget_password(email)

    def _render_sign_up(self):
        """Render sign-up form"""
        st.title("Sign Up")
        email = st.text_input("Email", key="email_signup")
        password = st.text_input("Password", type="password", key="password_signup")
        
        if st.button("Sign Up"):
            try:
                user = sign_up_with_email(email, password)
                if user:
                    st.session_state.current_page = "Live Monitoring"
                    st.rerun()
            except Exception as e:
                st.error(f"Sign up failed: {str(e)}")

    def _render_auth_screen(self):
        """Render authentication screen"""
        # Create three columns with the middle one containing the auth form
        left_col, center_col, right_col = st.columns([1, 2, 1])
        
        with center_col:
            # Add some vertical spacing
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Create a container for the auth form
            with st.container():
                # Title with custom styling
                st.markdown(
                    """
                    <h1 style='text-align: center; color: #1f77b4;'>
                        EXAM VISIO PRO
                    </h1>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Create tabs for Sign In and Sign Up
                tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
                
                with tab1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    email = st.text_input("Email", key="email_signin")
                    password = st.text_input("Password", type="password", key="password_signin")
                    
                    # Center the sign in button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("Sign In", use_container_width=True):
                            try:
                                user = sign_in_with_email(email, password)
                                if user:
                                    # Check email verification using user_metadata
                                    if not user.user_metadata.get('email_verified', False):
                                        st.error("Please verify your email address before signing in. Check your inbox for the verification link.")
                                        return
                                    st.session_state.current_page = "Live Monitoring"
                                    st.session_state.user = {
                                        'email': user.email,
                                        'id': user.id,
                                        'role': user.role
                                    }
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Sign in failed: {str(e)}")
                    
                    # Forgot password section
                    st.markdown("<br>", unsafe_allow_html=True)
                    email_forgot = st.text_input("Email for password reset", key="email_forgot")
                    if st.button("Forgot Password?", use_container_width=True):
                        forget_password(email_forgot)
                
                with tab2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    email = st.text_input("Email", key="email_signup")
                    password = st.text_input("Password", type="password", key="password_signup")
                    
                    # Center the sign up button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("Sign Up", use_container_width=True):
                            try:
                                user = sign_up_with_email(email, password)
                                if user:
                                    st.success("Sign up successful! Please check your email to verify your account.")
                                    st.info("You will need to verify your email before you can sign in.")
                                    time.sleep(2)
                                    st.rerun()
                            except Exception as e:
                                if "User already registered" in str(e):
                                    st.error("This email is already registered. Please sign in instead.")
                                else:
                                    st.error(f"Sign up failed: {str(e)}")
            
            # Add some spacing at the bottom
            st.markdown("<br><br>", unsafe_allow_html=True)

    def _start_monitoring(self, source_type):
        """Start video monitoring"""
        video_source = 0  # Default to webcam
        
        if source_type == "File":
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_source = tmp_file.name
        
        if st.session_state.video_processor.start_monitoring(video_source):
            st.success("Monitoring started")
            st.rerun()
        else:
            st.error("Failed to start monitoring")

    def _stop_monitoring(self):
        """Stop video monitoring"""
        if st.session_state.video_processor.stop_monitoring():
            st.warning("Monitoring stopped")
            st.rerun()
        else:
            st.error("Failed to stop monitoring")

    def _render_video_feed(self):
        """Display live video feed"""
        if st.session_state.video_processor.is_monitoring():
            frame_placeholder = st.empty()
            
            while st.session_state.video_processor.is_monitoring():
                frame = st.session_state.video_processor.get_latest_frame()
                if frame is not None:
                    # Run detection
                    results = self.model_manager.detect(frame)
                    counts = self.data_manager.update_detections(results)
                    
                    # Check for alerts
                    if (counts['cheating'] > 0 and 
                        st.session_state.video_processor.should_trigger_alert()):
                        alert_data = self.alert_manager.create_alert(frame, results)
                        self.data_manager.add_alert(alert_data)
                        
                        if st.session_state.email_alerts:
                            threading.Thread(
                                target=self.alert_manager.send_email_alert,
                                args=(alert_data,),
                                daemon=True
                            ).start()
                        
                        st.session_state.video_processor.last_alert_time = time.time()
                    
                    # Display frame
                    annotated_frame = self.alert_manager._annotate_frame(frame.copy(), results)
                    frame_placeholder.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB"
                    )
                
                time.sleep(1/Config.TARGET_FPS)
        else:
            st.info("Monitoring not active")

            
    def _update_metrics(self, counts, people_count):
        """Update the sidebar metrics"""
        self.cheating_metric.metric(
            "Cheating Detections", 
            st.session_state.detection_counts['cheating'])
        self.good_metric.metric(
            "Good Behavior", 
            st.session_state.detection_counts['good'])
        self.alerts_metric.metric(
            "Active Alerts", 
            len(st.session_state.alert_history))
        self.people_metric.metric(
            "People in Frame", 
            people_count)

    def _render_analytics(self):
        """Render the analytics dashboard"""
        st.title("📊 Analytics Dashboard")
        
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
        tab1, tab2, tab3, tab4 = st.tabs(["Counts", "Timeline", "Heatmap", "Proportions"])

        with tab1:
            fig, ax = plt.subplots()
            sns.barplot(
                x=['Cheating', 'Good'], 
                y=[st.session_state.detection_counts['cheating'], 
                   st.session_state.detection_counts['good']],
                hue=['Cheating', 'Good'],
                palette=['red', 'green'],
                legend=False,
                ax=ax
            )
            ax.set_title('Behavior Detection Counts')
            st.pyplot(fig)
            
        with tab2:
          if filtered.empty:
            st.warning("No detection data available for the selected time window")
          else:
            resampled = filtered.set_index('timestamp').resample('30s').sum()
            if resampled.empty:
                st.warning("Not enough data points to display timeline")
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                resampled.plot(ax=ax)
                ax.set_title('Detection Timeline')
                ax.set_ylabel('Detections per 30 Seconds')
                st.pyplot(fig)

        with tab3:
            if filtered.empty:
                st.warning("No detection data available for the selected time window")
            else:
                heatmap_data = filtered.pivot_table(
                    index=filtered['timestamp'].dt.date, 
                    columns=filtered['timestamp'].dt.hour, 
                    values='cheating', 
                    aggfunc='sum', 
                    fill_value=0)
                
                # Handle future warning for fillna
                heatmap_data = heatmap_data.infer_objects(copy=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt='d', ax=ax)
                ax.set_title('Cheating Heatmap by Date and Hour')
                st.pyplot(fig)        
                
        with tab4:
            fig, ax = plt.subplots()
            ax.pie(
                [st.session_state.detection_counts['cheating'], 
                 st.session_state.detection_counts['good']],
                labels=['Cheating', 'Good'], 
                autopct='%1.1f%%', startangle=90,
                colors=['red', 'green'], explode=(0.1, 0)
            )
            ax.axis('equal') 
            ax.set_title('Behavior Proportions')
            st.pyplot(fig)

    def _render_evidence(self):
        """Render the evidence review page"""
        st.title("🖼️ Evidence Gallery")
        
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

    def _render_alerts(self):
        """Alerts management page"""
        st.title("🚨 Alerts")
        
        if not st.session_state.alert_history:
            st.info("No alerts recorded")
            return
            
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            severity = st.select_slider(
                "Severity",
                options=["Low", "Medium", "High"],
                value="Medium"
            )
        with col2:
            date_filter = st.date_input("Filter by date")
        
        # Filter alerts
        min_conf = 0.5 if severity == "Low" else 0.7 if severity == "Medium" else 0.9
        filtered = [
            alert for alert in st.session_state.alert_history
            if alert['confidence'] >= min_conf and
            (not date_filter or alert['timestamp'].date() == date_filter)
        ]
        
        if not filtered:
            st.warning("No alerts match filters")
            return
            
        # Display alerts
        for alert in reversed(filtered):
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
                            file_name=os.path.basename(alert['image_path'])
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
        
        if not st.session_state.detection_history.empty:
            unique_dates = len(st.session_state.detection_history['timestamp'].dt.date.unique())
            avg_cheating = st.session_state.detection_history['cheating'].mean()
        else:
            unique_dates = 0
            avg_cheating = 0.0
            
        col1.metric("Total Sessions", unique_dates)
        col2.metric("Total Alerts", len(st.session_state.alert_history))
        col3.metric("Avg Cheating per Session", f"{avg_cheating:.1f}")
        
        # Date range filter
        st.subheader("Detailed History")
        if not st.session_state.detection_history.empty:
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
        else:
            st.info("No history data available yet.")
            

    def _render_settings(self):
        """System settings page"""
        st.title("⚙️ System Settings")
        
        # Video settings
        st.subheader("Video Configuration")
        col1, col2 = st.columns(2)
        with col1:
            new_width = st.number_input("Frame Width", 320, 1920, Config.FRAME_SIZE[0])
        with col2:
            new_height = st.number_input("Frame Height", 240, 1080, Config.FRAME_SIZE[1])
        
        if st.button("Apply Resolution"):
            Config.FRAME_SIZE = (new_width, new_height)
            st.success("Resolution updated")
        
        # Alert settings
        st.subheader("Alert Settings")
        st.session_state.email_alerts = st.checkbox(
            "Enable Email Alerts",
            value=st.session_state.email_alerts
        )
        Config.ALERT_COOLDOWN = st.number_input(
            "Alert Cooldown (seconds)",
            1, 60, Config.ALERT_COOLDOWN
        )
        
        # System info
        st.subheader("System Information")
        st.write(f"OpenCV Version: {cv2.__version__}")
        st.write(f"PyTorch Version: {torch.__version__}")
        st.write(f"Streamlit Version: {st.__version__}")
        
        # Diagnostics
        if st.button("Run Diagnostics"):
            self._run_diagnostics()

    def _run_diagnostics(self):
        """Run system diagnostics"""
        st.subheader("Diagnostics Results")
        
        # Camera check
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                st.success("✅ Camera accessible")
                cap.release()
            else:
                st.error("❌ Camera not accessible")
        except Exception as e:
            st.error(f"❌ Camera check failed: {e}")
        
        # Model check
        try:
            test_model = YOLO(Config.MODEL_PATH)
            st.success("✅ Model loaded successfully")
        except Exception as e:
            st.error(f"❌ Model load failed: {e}")
        
        # Filesystem check
        try:
            test_file = os.path.join(Config.EVIDENCE_DIR, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.unlink(test_file)
            st.success("✅ Filesystem writable")
        except Exception as e:
            st.error(f"❌ Filesystem error: {e}")

# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    app = ExamVisioUI()
    app.run()