import os
import cv2
from streamlit_option_menu import option_menu
import time
import queue
import streamlit as st
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
from supabase import create_client, Client
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set in .env file")

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
        self._prev_frame = None
        self._motion_level = 0.0
        self._motion_threshold = 25.0  # Threshold for motion detection

    def get_motion_level(self) -> float:
        """Get the current motion level normalized between 0.0 and 1.0"""
        return min(1.0, self._motion_level / self._motion_threshold)

    def _compute_motion_level(self, current_frame):
        """Compute motion level between consecutive frames"""
        if self._prev_frame is None:
            self._prev_frame = current_frame
            return
        
        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        
        # Apply threshold to get significant changes
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion level as percentage of changed pixels
        motion = np.mean(thresh > 0) * 100
        
        # Update motion level with smoothing
        self._motion_level = (0.7 * self._motion_level) + (0.3 * motion)
        
        # Update previous frame
        self._prev_frame = current_frame

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
                            try:
                                # Compute motion level
                                self._compute_motion_level(frame)
                                
                                # Queue frame
                                if self.frame_queue.full():
                                    self.frame_queue.get_nowait()  # Discard oldest
                                self.frame_queue.put_nowait(frame)
                            except Exception as proc_error:
                                logger.error(f"Frame processing error: {proc_error}")
                            
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
            # Compute motion level
            self._compute_motion_level(frame)
            
            # Queue frame
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
        if 'video_processor' not in st.session_state or not hasattr(st.session_state.video_processor, 'get_motion_level'):
            st.session_state.video_processor = VideoProcessor()
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Exam Configuration"
            
        # Initialize notification system
        from notification_manager import NotificationManager
        self.notification_manager = NotificationManager()
        
        # Initialize incident analysis state
        if 'incident_patterns' not in st.session_state:
            st.session_state.incident_patterns = {
                'frequent_locations': {},  # Track common incident locations
                'time_patterns': {},      # Track time-based patterns
                'severity_levels': {},    # Track severity distributions
                'repeat_incidents': {}    # Track repeat incidents
            }
            
        # Initialize Supabase client
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        from sync_manager import SyncManager
        if 'sync_manager' not in st.session_state:
            st.session_state.sync_manager = SyncManager(self.supabase)
            
        # Import exam management functions
        from exam_configuration import configure_examination
        from generate_report import (
            schedule_exam,
            generate_report
        )
        # Store the functions as instance variables
        self.configure_examination = configure_examination
        self.schedule_exam = schedule_exam
        self.generate_report = generate_report

        # Store the pages dictionary
        self.pages = {
            "Dashboard": self._render_dashboard,
            "Exam Configuration": self.configure_examination,
            "Exam Scheduling": self.schedule_exam,
            "Live Monitoring": self._render_live_monitoring,
            "Analytics Dashboard": self._render_analytics,
            "Evidence Review": self._render_evidence,
            "Reports": self.generate_report,
            "System Settings": self._render_settings
        }
        
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
        with st.sidebar:
            st.title("📚 ExamVisio Pro")
            st.markdown(f"### 👤 Welcome, {st.session_state.get('name', 'Admin')}")
            
            # Initialize navigation state if not exists
            if 'current_page' not in st.session_state:
                st.session_state.current_page = "Dashboard"
            
            # Map the old page names to new option menu names
            page_mapping = {
                "Exam Configuration": "🛠️ Exam Configuration",
                "Exam Scheduling": "📅 Exam Scheduling",
                "Live Monitoring": "📡 Live Monitoring",
                "Dashboard": "📊 Dashboard",
                "Analytics Dashboard": "📈 Analytics Dashboard",
                "Evidence Review": "🕵️ Evidence Review",
                "Reports": "📄 Reports",
                "System Settings": "⚙️ System Settings"
            }
            
            # Get the current index for the option menu
            current_page = st.session_state.current_page
            default_index = list(page_mapping.values()).index(page_mapping.get(current_page, "📊 Dashboard"))
            
            selected = option_menu(
                menu_title="Navigation",
                options=[
                    "🛠️ Exam Configuration",
                    "📅 Exam Scheduling",
                    "📡 Live Monitoring",
                    "📊 Dashboard",
                    "📈 Analytics Dashboard",
                    "🕵️ Evidence Review",
                    "📄 Reports",
                    "⚙️ System Settings"
                ],
                icons=[
                    "tools",
                    "calendar",
                    "camera-video",
                    "speedometer",
                    "bar-chart-line",
                    "search",
                    "file-earmark-text",
                    "gear"
                ],
                default_index=default_index,
                styles={
                    "container": {"padding": "0!important"},
                    "nav-link": {"font-size": "15px", "margin": "3px 0"},
                    "nav-link-selected": {"background-color": "#0e6efd", "color": "white"},
                }
            )
            
            # Reverse mapping to convert option menu selection to original page names
            reverse_mapping = {v: k for k, v in page_mapping.items()}
            st.session_state.current_page = reverse_mapping.get(selected, selected)
            
            # Detection settings
            st.markdown("---")
            st.markdown("### Detection Settings")
            Config.CONFIDENCE_THRESHOLD = st.slider(
                "Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
            Config.ALERT_CONFIDENCE = st.slider(
                "Alert Threshold", 0.1, 0.9, 0.6, 0.05)
            st.session_state.email_alerts = st.checkbox(
                "Email Alerts", value=st.session_state.email_alerts)

            # Metrics display
            st.markdown("---")
            st.markdown("### 📊 Live Metrics")
            self.cheating_metric = st.empty()
            self.good_metric = st.empty()
            self.alerts_metric = st.empty()
            self.people_metric = st.empty()
            
            # Sign Out Button
            st.markdown("---")
            if st.button("🚪 Sign Out"):
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

    def _render_dashboard(self):
        """Render the main dashboard"""
        st.title("📊 ExamVisio Pro Dashboard")
        
        # Initialize state
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Use the existing Supabase client
        supabase = self.supabase
        
        # Check for local monitoring data
        has_local_data = (hasattr(st.session_state, 'detection_history') and 
                         isinstance(getattr(st.session_state, 'detection_history', None), pd.DataFrame) and 
                         not getattr(st.session_state, 'detection_history', pd.DataFrame()).empty)
        
            # Define DataContainer class outside try-except
        class DataContainer:
            def __init__(self, data):
                self.data = data

        # Try to fetch database statistics
        try:
            exams_response = supabase.table("exams").select("*").execute()
            detections_response = supabase.table("detections").select("*").execute()
            alerts_response = supabase.table("alerts").select("*").execute()
            
            # Safely get data or default to empty lists
            exams_data = getattr(exams_response, 'data', [])
            detections_data = getattr(detections_response, 'data', [])
            alerts_data = getattr(alerts_response, 'data', [])
            
            has_db_data = bool(exams_data)
            
            # Create data containers
            exams = DataContainer(exams_data)
            detections = DataContainer(detections_data)
            alerts = DataContainer(alerts_data)
            
        except Exception as e:
            logger.error(f"Database fetch failed: {str(e)}")
            has_db_data = False
            exams = DataContainer([])
            detections = DataContainer([])
            alerts = DataContainer([])        # Show appropriate content based on data availability
        if not has_local_data and not has_db_data:
            st.info("Welcome to ExamVisio Pro! To get started:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("➕ Configure New Exam", use_container_width=True):
                    st.session_state.current_page = "Exam Configuration"
                    st.rerun()
            with col2:
                if st.button("📅 Schedule an Exam", use_container_width=True):
                    st.session_state.current_page = "Exam Scheduling"
                    st.rerun()
            with col3:
                if st.button("📡 Start Monitoring", use_container_width=True):
                    st.session_state.current_page = "Live Monitoring"
                    st.rerun()
                    
            # Show getting started guide
            st.markdown("""
            ### Getting Started Guide
            1. **Configure Exam**: Set up exam details, course information, and monitoring settings
            2. **Schedule Exam**: Plan your exam sessions and set monitoring duration
            3. **Start Monitoring**: Begin real-time exam supervision
            
            Once you start monitoring or configure exams, you'll see:
            - Real-time monitoring statistics
            - Exam session analytics
            - Detection reports and alerts
            """)
            return
            
        # Show local monitoring data if available
        if has_local_data:
            st.subheader("📊 Current Monitoring Session")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Detections", len(st.session_state.detection_history))
            with col2:
                st.metric("Alerts", len(st.session_state.alert_history))
            with col3:
                cheating_count = st.session_state.detection_counts.get('cheating', 0)
                st.metric("Cheating Incidents", cheating_count)
            with col4:
                good_count = st.session_state.detection_counts.get('good', 0)
                st.metric("Normal Behavior", good_count)
                
        # Show database statistics if available
        if has_db_data:
            create_tables_sql = """
                    -- Create exams table
                    CREATE TABLE IF NOT EXISTS public.exams (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        exam_name TEXT NOT NULL,
                        exam_type TEXT NOT NULL,
                        course_code TEXT NOT NULL,
                        department TEXT NOT NULL,
                        instructor TEXT NOT NULL,
                        degree_type TEXT NOT NULL,
                        year_of_study TEXT NOT NULL,
                        total_students INTEGER NOT NULL DEFAULT 0,
                        start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        end_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        duration INTEGER NOT NULL,  -- Duration in minutes
                        status TEXT NOT NULL DEFAULT 'scheduled',
                        venue TEXT NOT NULL,
                        face_detection BOOLEAN NOT NULL DEFAULT true,
                        noise_detection BOOLEAN NOT NULL DEFAULT true,
                        multi_face_detection BOOLEAN NOT NULL DEFAULT true,
                        confidence_threshold FLOAT NOT NULL DEFAULT 0.5,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    -- Create detections table
                    CREATE TABLE IF NOT EXISTS public.detections (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        exam_id UUID REFERENCES public.exams(id),
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        confidence FLOAT NOT NULL,
                        people_count INTEGER NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    -- Create alerts table
                    CREATE TABLE IF NOT EXISTS public.alerts (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        exam_id UUID REFERENCES public.exams(id),
                        detection_id UUID REFERENCES public.detections(id),
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        alert_type TEXT NOT NULL,
                        confidence FLOAT NOT NULL,
                        evidence_path TEXT,
                        reviewed BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    -- Add indexes for better performance
                    CREATE INDEX IF NOT EXISTS idx_exams_status ON public.exams(status);
                    CREATE INDEX IF NOT EXISTS idx_detections_exam_id ON public.detections(exam_id);
                    CREATE INDEX IF NOT EXISTS idx_alerts_exam_id ON public.alerts(exam_id);
                    CREATE INDEX IF NOT EXISTS idx_alerts_reviewed ON public.alerts(reviewed);
                    """
            try:
                # Execute the SQL to create tables
                supabase.raw(create_tables_sql).execute()
                
                # Insert sample data
                # Calculate duration in minutes
                start_time = datetime.now()
                end_time = start_time + timedelta(hours=1)
                duration = int((end_time - start_time).total_seconds() / 60)

                exam_data = {
                    "exam_name": "Test Exam",
                    "exam_type": "Quiz",
                    "course_code": "TEST101",
                    "department": "Test Department",
                    "instructor": "Test Instructor",
                    "degree_type": "BSc",
                    "year_of_study": "1st Year",
                    "total_students": 0,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": duration,  # Duration in minutes
                    "status": "scheduled",
                    "venue": "Test Room",
                    "face_detection": True,
                    "noise_detection": True,
                    "multi_face_detection": True,
                    "confidence_threshold": 0.5
                }
                result = supabase.table("exams").insert(exam_data).execute()
                exam_id = result.data[0]['id']
            
                # Insert sample detection
                detection_data = {
                    "exam_id": exam_id,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.0,
                    "people_count": 0
                }
                result = supabase.table("detections").insert(detection_data).execute()
                detection_id = result.data[0]['id']
                
                # Insert sample alert
                alert_data = {
                    "exam_id": exam_id,
                    "detection_id": detection_id,
                    "timestamp": datetime.now().isoformat(),
                    "alert_type": "test",
                    "confidence": 0.0,
                    "evidence_path": "",
                    "reviewed": False
                }
                supabase.table("alerts").insert(alert_data).execute()
                
                st.success("Database tables created successfully!")
                st.rerun()
            except Exception as setup_error:
                st.error(f"Failed to create tables: {str(setup_error)}")
                return
            
        if exams.data:
            total_exams = len(exams.data)
            scheduled = len([e for e in exams.data if e['status'] == 'scheduled'])
            running = len([e for e in exams.data if e['status'] == 'running'])
            completed = len([e for e in exams.data if e['status'] == 'completed'])
            
            # Calculate additional metrics
            total_monitored_time = sum([
                (datetime.fromisoformat(e['end_time']) - datetime.fromisoformat(e['start_time'])).total_seconds() / 3600 
                for e in exams.data if e['status'] == 'completed'
            ])
            total_students = sum(e['total_students'] for e in exams.data)
            alert_rate = len(alerts.data) / total_exams if total_exams > 0 else 0
            
            # Add attendance data
            attendance_data = []
            for exam in exams.data:
                if exam['status'] == 'completed':
                    detections_in_exam = [d for d in detections.data if d['exam_id'] == exam['id']]
                    unique_faces = len(set(d['people_count'] for d in detections_in_exam))
                    attendance_data.append({
                        'exam': exam['exam_name'],
                        'expected': exam['total_students'],
                        'actual': unique_faces,
                        'percentage': (unique_faces / exam['total_students'] * 100) if exam['total_students'] > 0 else 0
                    })
            
            # Display metrics in two rows
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Exams", total_exams)
            with col2:
                st.metric("Scheduled", scheduled)
            with col3:
                st.metric("Running", running, delta="+1" if running > 0 else None)
            with col4:
                st.metric("Completed", completed)
                
            # Show attendance statistics if available
            if attendance_data:
                st.subheader("📊 Attendance Statistics")
                df = pd.DataFrame(attendance_data)
                
                # Calculate average attendance rate
                avg_attendance = df['percentage'].mean()
                st.metric("Average Attendance Rate", f"{avg_attendance:.1f}%")
                
                # Show attendance table
                st.dataframe(
                    df,
                    column_config={
                        "exam": "Exam",
                        "expected": "Expected",
                        "actual": "Actual",
                        "percentage": st.column_config.ProgressColumn(
                            "Attendance Rate",
                            help="Percentage of expected students present",
                            format="%{:.1f}",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    use_container_width=True
                )
            
            # Additional metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", total_students)
            with col2:
                st.metric("Hours Monitored", f"{total_monitored_time:.1f}")
            with col3:
                st.metric("Alerts/Exam", f"{alert_rate:.2f}")
            with col4:
                st.metric("Active Alerts", len([a for a in alerts.data if not a['reviewed']]))
            
            # Quick Actions
            st.subheader("🔄 Quick Actions")
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            with quick_col1:
                if st.button("➕ Configure New Exam", use_container_width=True):
                    st.session_state.current_page = "Exam Configuration"
                    st.rerun()
            with quick_col2:
                if st.button("📅 View Schedule", use_container_width=True):
                    st.session_state.current_page = "Exam Scheduling"
                    st.rerun()
            with quick_col3:
                if st.button("📊 Generate Reports", use_container_width=True):
                    st.session_state.current_page = "Reports"
                    st.rerun()
            
            # Recent and Upcoming Exams
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📅 Upcoming Exams")
                upcoming = [e for e in exams.data if e['status'] == 'scheduled']
                upcoming.sort(key=lambda x: x['start_time'])
                if upcoming:
                    for exam in upcoming[:3]:
                        with st.expander(f"{exam['exam_name']} - {exam['department']}"):
                            st.write(f"📆 Date: {exam['start_time'].split('T')[0]}")
                            st.write(f"🕒 Time: {exam['start_time'].split('T')[1][:5]}")
                            st.write(f"👨‍🏫 Instructor: {exam['instructor']}")
                            if st.button("Start Monitoring", key=f"start_{exam['id']}"):
                                st.session_state.current_page = "Live Monitoring"
                                st.session_state.current_exam_id = exam['id']
                                st.rerun()
                else:
                    st.info("No upcoming exams scheduled")
            
            with col2:
                st.subheader("📝 Recent Reports")
                completed_exams = [e for e in exams.data if e['status'] == 'completed']
                completed_exams.sort(key=lambda x: x['end_time'], reverse=True)
                if completed_exams:
                    for exam in completed_exams[:3]:
                        with st.expander(f"{exam['exam_name']} - {exam['department']}"):
                            st.write(f"📆 Date: {exam['end_time'].split('T')[0]}")
                            st.write(f"👨‍🏫 Instructor: {exam['instructor']}")
                            if st.button("View Report", key=f"report_{exam['id']}"):
                                st.session_state.current_page = "Reports"
                                st.session_state.selected_exam_id = exam['id']
                                st.rerun()
                else:
                    st.info("No completed exams yet")
        else:
            st.info("Welcome to ExamVisio Pro! Start by configuring your first exam.")
            if st.button("Configure New Exam"):
                st.session_state.current_page = "Exam Configuration"
                st.rerun()
    
    def _render_current_page(self):
        """Render the current active page"""
        self.pages[st.session_state.current_page]()

    def _render_live_monitoring(self):
        """Live monitoring page"""
        st.title("🎥 Live Monitoring")
        
        # Initialize behavior analyzer if not exists
        if 'behavior_analyzer' not in st.session_state:
            from behavior_analyzer import BehaviorAnalyzer
            st.session_state.behavior_analyzer = BehaviorAnalyzer()
            
        # Proctor Control Panel
        with st.expander("🎛️ Proctor Control Panel", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🎥 Camera Controls")
                camera_mode = st.selectbox(
                    "Camera Mode",
                    ["Standard", "High Resolution", "Low Light", "Motion Focus"]
                )
                st.checkbox("Enable Audio Monitoring", key="audio_monitoring")
                st.checkbox("Enable Motion Detection", key="motion_detection")
                
            with col2:
                st.subheader("🚨 Alert Settings")
                alert_sensitivity = st.slider("Alert Sensitivity", 0.0, 1.0, 0.6)
                min_incident_duration = st.number_input("Min Incident Duration (s)", 1, 60, 3)
                consecutive_alerts = st.number_input("Consecutive Alerts for Action", 1, 10, 3)
                
            with col3:
                st.subheader("🤖 AI Assistant")
                st.checkbox("Enable Smart Tracking", key="smart_tracking")
                st.checkbox("Auto-zoom on Incidents", key="auto_zoom")
                behavior_threshold = st.slider("Behavior Analysis Threshold", 0.0, 1.0, 0.7)
        
        # Session info
        if 'current_exam_id' in st.session_state:
            exam = self.supabase.table("exams").select("*").eq('id', st.session_state.current_exam_id).execute()
            if exam.data:
                exam_data = exam.data[0]
                st.markdown(f"""
                    ### 📝 Current Session
                    - **Exam**: {exam_data['exam_name']}
                    - **Course**: {exam_data['course_code']}
                    - **Department**: {exam_data['department']}
                    - **Instructor**: {exam_data['instructor']}
                    - **Students**: {exam_data['total_students']}
                    """)
                
                # Session metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Session Duration", f"{(datetime.now() - datetime.fromisoformat(exam_data['start_time'])).seconds // 60}m")
                with col2:
                    st.metric("Detections", len(st.session_state.detection_history))
                with col3:
                    st.metric("Alerts", len(st.session_state.alert_history))
                with col4:
                    st.metric("Alert Rate", f"{len(st.session_state.alert_history) / max(1, len(st.session_state.detection_history)):.2%}")
                
                # Add exam progress
                exam_start = datetime.fromisoformat(exam_data['start_time'])
                exam_end = datetime.fromisoformat(exam_data['end_time'])
                total_duration = (exam_end - exam_start).total_seconds() / 60
                elapsed_duration = (datetime.now() - exam_start).total_seconds() / 60
                progress = min(100, (elapsed_duration / total_duration) * 100)
                
                # Display progress bar
                st.progress(progress / 100)
                st.write(f"⏱️ Time Remaining: {max(0, int(total_duration - elapsed_duration))} minutes")
                
                # Status indicator
                status_col1, status_col2 = st.columns([1, 3])
                with status_col1:
                    if progress >= 100:
                        st.error("🔴 Exam Ended")
                    elif len(st.session_state.alert_history) > 0:
                        st.warning("🟡 Alerts Detected")
                    else:
                        st.success("🟢 Normal Operation")
                with status_col2:
                    st.write(f"Current Status: {'High Alert' if len(st.session_state.alert_history) > 5 else 'Moderate Alert' if len(st.session_state.alert_history) > 0 else 'Normal'}")
        
        # Video source selection
        source_type = st.radio("Video Source", ["Webcam", "File"])
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Monitoring", use_container_width=True):
                self._start_monitoring(source_type)
        with col2:
            if st.button("Stop Monitoring", use_container_width=True):
                self._stop_monitoring()
        with col3:
            if st.button("End Session", type="primary", use_container_width=True):
                if 'current_exam_id' in st.session_state:
                    self.supabase.table("exams").update(
                        {"status": "completed", "end_time": datetime.now().isoformat()}
                    ).eq('id', st.session_state.current_exam_id).execute()
                    st.success("Exam session completed")
                    time.sleep(2)
                    st.session_state.pop('current_exam_id', None)
                    st.rerun()
        
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
            
            # Initialize analyzers if not exists
            if 'anomaly_detector' not in st.session_state:
                from anomaly_detector import AnomalyDetector
                st.session_state.anomaly_detector = AnomalyDetector()
            
            while st.session_state.video_processor.is_monitoring():
                frame = st.session_state.video_processor.get_latest_frame()
                if frame is not None:
                    # Run detection
                    results = self.model_manager.detect(frame)
                    counts = self.data_manager.update_detections(results)
                    
                    # Prepare detection data for analysis
                    # Extract confidence values from boxes in results
                    confidence_values = []
                    for det in results:
                        if det.boxes is not None:
                            for box in det.boxes:
                                confidence_values.append(float(box.conf))
                    
                    current_data = {
                        'confidence': max(confidence_values) if confidence_values else 0.0,
                        'people_count': sum(len(det.boxes) for det in results if det.boxes is not None),
                        'motion_level': st.session_state.video_processor.get_motion_level(),
                        'alert_frequency': len(st.session_state.alert_history) / max(1, len(st.session_state.detection_history)),
                        'duration': 0  # Will be updated if alert is created
                    }
                    
                    # Run anomaly detection
                    anomaly_result = st.session_state.anomaly_detector.detect_anomalies(current_data)
                    
                    # Run behavior analysis
                    behavior_analysis = st.session_state.behavior_analyzer.analyze_behavior_patterns(
                        st.session_state.detection_history)
                    
                    # Check for alerts with enhanced criteria
                    if ((counts['cheating'] > 0 and 
                         st.session_state.video_processor.should_trigger_alert()) or
                        anomaly_result['is_anomaly']):
                        
                        alert_data = self.alert_manager.create_alert(frame, results)
                        # Add anomaly and behavior analysis data
                        alert_data.update({
                            'anomaly_score': anomaly_result['score'],
                            'anomaly_severity': anomaly_result['severity'],
                            'behavior_risk_score': behavior_analysis.get('risk_score', 0),
                            'insights': anomaly_result['insights']
                        })
                        self.data_manager.add_alert(alert_data)
                        
                        # Queue alert for syncing
                        if 'current_exam_id' in st.session_state:
                            sync_alert_data = {
                                'exam_id': st.session_state.current_exam_id,
                                'detection_id': alert_data['id'],
                                'timestamp': alert_data['timestamp'].isoformat(),
                                'alert_type': 'cheating',
                                'confidence': float(alert_data['confidence']),
                                'evidence_path': alert_data['image_path'],
                                'reviewed': False
                            }
                            st.session_state.sync_manager.queue_alert(sync_alert_data)
                            
                            # Add notification with severity level
                            severity = "error" if alert_data['confidence'] > 0.8 else "warning"
                            self.notification_manager.add_notification(
                                f"🚨 Cheating detected with {alert_data['confidence']:.0%} confidence",
                                level=severity,
                                data={
                                    'exam_id': st.session_state.current_exam_id,
                                    'alert_id': alert_data['id'],
                                    'confidence': alert_data['confidence']
                                }
                            )
                            
                            # Update incident patterns
                            hour = alert_data['timestamp'].hour
                            location = f"Camera {st.session_state.video_processor.get_camera_id()}"
                            
                            # Update time patterns
                            st.session_state.incident_patterns['time_patterns'][hour] = \
                                st.session_state.incident_patterns['time_patterns'].get(hour, 0) + 1
                                
                            # Update location patterns
                            st.session_state.incident_patterns['frequent_locations'][location] = \
                                st.session_state.incident_patterns['frequent_locations'].get(location, 0) + 1
                        
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
            
        # Add Incident Analysis Section
        st.subheader("🔍 Incident Pattern Analysis")
        
        if len(st.session_state.alert_history) > 0:
            # Analyze patterns
            current_alerts = st.session_state.alert_history
            
            # Time pattern analysis
            time_periods = {
                'morning': 0,
                'afternoon': 0,
                'evening': 0
            }
            
            for alert in current_alerts:
                hour = alert['timestamp'].hour
                if 5 <= hour < 12:
                    time_periods['morning'] += 1
                elif 12 <= hour < 17:
                    time_periods['afternoon'] += 1
                else:
                    time_periods['evening'] += 1
            
            # Display patterns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⏰ Time Patterns")
                fig, ax = plt.subplots()
                times = list(time_periods.keys())
                values = list(time_periods.values())
                ax.bar(times, values)
                ax.set_title('Incidents by Time of Day')
                st.pyplot(fig)
                
                # Add insights
                max_period = max(time_periods.items(), key=lambda x: x[1])[0]
                st.info(f"📊 Most incidents occur during the {max_period}")
            
            with col2:
                st.subheader("📈 Severity Analysis")
                severity_levels = {
                    'High': len([a for a in current_alerts if a['confidence'] > 0.8]),
                    'Medium': len([a for a in current_alerts if 0.6 <= a['confidence'] <= 0.8]),
                    'Low': len([a for a in current_alerts if a['confidence'] < 0.6])
                }
                
                fig, ax = plt.subplots()
                ax.pie(
                    severity_levels.values(),
                    labels=severity_levels.keys(),
                    autopct='%1.1f%%',
                    colors=['red', 'orange', 'yellow']
                )
                ax.set_title('Incident Severity Distribution')
                st.pyplot(fig)
            
            # Trend Analysis
            st.subheader("📊 Trend Analysis")
            alert_counts = {}
            for alert in current_alerts:
                date = alert['timestamp'].date()
                alert_counts[date] = alert_counts.get(date, 0) + 1
            
            if len(alert_counts) > 1:
                dates = list(alert_counts.keys())
                counts = list(alert_counts.values())
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(dates, counts, marker='o')
                ax.set_title('Incident Trends Over Time')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Incidents')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Calculate trend
                if counts[-1] > counts[0]:
                    st.warning("⚠️ Incident rate is increasing")
                elif counts[-1] < counts[0]:
                    st.success("✅ Incident rate is decreasing")
                else:
                    st.info("ℹ️ Incident rate is stable")
        else:
            st.info("No incident data available for analysis")

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
            
        # Database connectivity check
        try:
            test_data = {'test': True, 'timestamp': datetime.now().isoformat()}
            response = self.supabase.table('exams').insert(test_data).execute()
            if response.data:
                self.supabase.table('exams').delete().eq('test', True).execute()
                st.success("✅ Database connection successful")
            else:
                st.error("❌ Database write failed")
        except Exception as e:
            st.error(f"❌ Database connection failed: {e}")
            
        # Check system resources
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            st.subheader("System Resources")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CPU Usage", f"{cpu_percent}%", 
                         delta_color="inverse",
                         delta="High" if cpu_percent > 80 else "Normal")
            with col2:
                st.metric("Memory Usage", f"{memory_percent}%",
                         delta_color="inverse",
                         delta="High" if memory_percent > 80 else "Normal")
            with col3:
                st.metric("Disk Usage", f"{disk_percent}%",
                         delta_color="inverse",
                         delta="High" if disk_percent > 80 else "Normal")
        except ImportError:
            st.info("💡 Install psutil for system resource monitoring")

# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    app = ExamVisioUI()
    app.run()