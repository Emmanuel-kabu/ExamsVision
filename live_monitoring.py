import streamlit as st
import cv2
import logging
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
import pytz
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
import time
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Exceptions
class MonitoringError(Exception):
    """Base class for monitoring exceptions"""
    pass

class CameraError(MonitoringError):
    """Camera-related exceptions"""
    pass

class DatabaseError(MonitoringError):
    """Database-related exceptions"""
    pass

@dataclass
class Detection:
    """Data class for detection results"""
    type: str
    confidence: float
    timestamp: datetime
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    extra_data: Optional[Dict[str, Any]] = None

class DetectionType(Enum):
    """Enum for detection types"""
    FACE = "face"
    OBJECT = "object"
    MOTION = "motion"
    PERSON = "person"
    PHONE = "phone"
    GOOD_BEHAVIOR = "good_behavior"

class ExamMonitoringSystem:
    """Handles real-time exam monitoring and alerts with enhanced features"""
    
    def __init__(self, db_manager, video_processor, alert_manager):
        """
        Initialize the monitoring system.
        
        Args:
            db_manager: Database manager instance
            video_processor: Video processing instance
            alert_manager: Alert manager instance
        """
        self.db_manager = db_manager
        self.video_processor = video_processor
        self.alert_manager = alert_manager
        self.monitoring_threads = {}
        self.frame_queues = {}
        self.stop_flags = {}
        self.camera_sources = {}
        self.lock = threading.Lock()  # For thread-safe operations

    # ======================
    # Public Interface
    # ======================
    
    def get_active_exam(self, exam_id: str) -> Optional[Dict]:
        """Get active exam details from database"""
        try:
            with self.lock:
                # Get scheduled exam
                scheduled = self.db_manager.supabase.table("scheduled_exams").select(
                    "*"
                ).eq("id", exam_id).single().execute()
                
                if not scheduled.data:
                    logger.warning(f"No scheduled exam found with ID {exam_id}")
                    return None
                
                # Get exam config
                config = self.db_manager.supabase.table("exams").select(
                    "*"
                ).eq("id", scheduled.data['exam_id']).single().execute()
                
                if not config.data:
                    logger.warning(f"No exam config found for ID {scheduled.data['exam_id']}")
                    return None
                    
                return {**config.data, **scheduled.data}
                
        except Exception as e:
            logger.error(f"Error getting active exam: {str(e)}")
            raise DatabaseError(f"Failed to get exam data: {str(e)}")
    
    def start_monitoring(self, exam_id: str, camera_source: int = 0) -> bool:
        """Start monitoring for a specific exam"""
        if exam_id in self.monitoring_threads:
            logger.warning(f"Monitoring already active for exam {exam_id}")
            return False
            
        exam_data = self.get_active_exam(exam_id)
        if not exam_data:
            logger.error(f"Could not find exam with ID {exam_id}")
            return False
            
        try:
            # Initialize camera
            cap = self._initialize_camera(camera_source)
            
            with self.lock:
                # Store camera source
                self.camera_sources[exam_id] = cap
                
                # Initialize frame queue and stop flag
                self.frame_queues[exam_id] = queue.Queue(maxsize=10)
                self.stop_flags[exam_id] = threading.Event()
                
                # Start monitoring thread
                monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    args=(exam_id, exam_data),
                    daemon=True,
                    name=f"MonitoringThread-{exam_id}"
                )
                self.monitoring_threads[exam_id] = monitor_thread
                monitor_thread.start()
                
            logger.info(f"Started monitoring for exam {exam_data['exam_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            self._cleanup_resources(exam_id)
            raise MonitoringError(f"Failed to start monitoring: {str(e)}")
    
    def stop_all_monitoring(self) -> None:
        """Stop all active monitoring sessions"""
        with self.lock:
            for exam_id in list(self.monitoring_threads.keys()):
                self.stop_monitoring(exam_id)
    
    def stop_monitoring(self, exam_id: str) -> None:
        """Stop monitoring for a specific exam"""
        if exam_id not in self.stop_flags:
            logger.warning(f"No monitoring session found for exam {exam_id}")
            return
            
        try:
            # Signal thread to stop
            self.stop_flags[exam_id].set()
            
            # Wait for thread to finish
            if exam_id in self.monitoring_threads:
                thread = self.monitoring_threads[exam_id]
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"Monitoring thread for exam {exam_id} did not stop gracefully")
                
            # Calculate and save metrics
            self._save_exam_metrics(exam_id)
            
            # Clean up resources
            self._cleanup_resources(exam_id)
            
            logger.info(f"Stopped monitoring for exam {exam_id}")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            raise MonitoringError(f"Failed to stop monitoring: {str(e)}")
    
    def get_monitoring_statistics(self, exam_id: str) -> Dict:
        """Get current monitoring statistics for an exam"""
        try:
            stats = self.alert_manager.get_exam_statistics(exam_id)
            return {
                "face_detections": stats.get(DetectionType.FACE.value, 0),
                "object_detections": stats.get(DetectionType.OBJECT.value, 0),
                "motion_detections": stats.get(DetectionType.MOTION.value, 0),
                "person_detections": stats.get(DetectionType.PERSON.value, 0),
                "phone_detections": stats.get(DetectionType.PHONE.value, 0),
                "good_behavior": stats.get(DetectionType.GOOD_BEHAVIOR.value, 0),
                "total_alerts": sum(stats.values())
            }
        except Exception as e:
            logger.error(f"Error getting monitoring statistics: {str(e)}")
            return {
                "face_detections": 0,
                "object_detections": 0,
                "motion_detections": 0,
                "person_detections": 0,
                "phone_detections": 0,
                "good_behavior": 0,
                "total_alerts": 0
            }

    # ======================
    # Monitoring Interface
    # ======================
    
    def render_monitoring_interface(self) -> None:
        """Render the monitoring interface in Streamlit"""
        st.title("🔍 Live Exam Monitoring Dashboard")
        
        try:
            # Get current and upcoming exams
            now = datetime.now(pytz.UTC)
            
            # Get scheduled exams that are happening now or in the future
            scheduled = self._get_scheduled_exams(now)
            
            if not scheduled.data:
                st.info("No active or upcoming exams found")
                return
                
            # Process exam data
            config_map = self._get_exam_configs(scheduled)
            current_exams, upcoming_exams = self._categorize_exams(scheduled, config_map, now)
            
            # Display current exams
            self._render_current_exams(current_exams, now)
            
            # Display upcoming exams
            self._render_upcoming_exams(upcoming_exams)
                
        except Exception as e:
            logger.error(f"Error in monitoring interface: {str(e)}")
            st.error("Failed to load monitoring interface")

    # ======================
    # Private Methods
    # ======================
    
    def _initialize_camera(self, camera_source: int) -> cv2.VideoCapture:
        """Initialize and verify camera connection"""
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise CameraError(f"Failed to open camera source {camera_source}")
        
        # Test camera read
        for _ in range(3):  # Try a few times
            ret, _ = cap.read()
            if ret:
                return cap
            time.sleep(0.1)
        
        cap.release()
        raise CameraError(f"Camera source {camera_source} not responding")

    def _cleanup_resources(self, exam_id: str) -> None:
        """Clean up resources for an exam"""
        with self.lock:
            # Release camera
            if exam_id in self.camera_sources:
                self.camera_sources[exam_id].release()
                del self.camera_sources[exam_id]
            
            # Remove thread reference
            if exam_id in self.monitoring_threads:
                del self.monitoring_threads[exam_id]
            
            # Clean up other resources
            if exam_id in self.stop_flags:
                del self.stop_flags[exam_id]
            if exam_id in self.frame_queues:
                del self.frame_queues[exam_id]

    def _monitoring_loop(self, exam_id: str, exam_data: Dict) -> None:
        """Main monitoring loop for an exam"""
        try:
            cap = self.camera_sources[exam_id]
            last_health_check = datetime.now()
            frame_count = 0
            
            while not self.stop_flags[exam_id].is_set():
                # Periodic camera health check
                if (datetime.now() - last_health_check) > timedelta(seconds=30):
                    if not self._check_camera_health(exam_id, cap):
                        break
                    last_health_check = datetime.now()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame (every 3rd frame for performance)
                frame_count += 1
                if frame_count % 3 == 0:
                    detections = self.video_processor.process_frame(frame)
                    
                    # Process detections based on settings
                    if detections:
                        settings = exam_data.get('monitoring_settings', {})
                        self._handle_detections(exam_id, frame, detections, settings)
                
                # Update frame queue for display
                if not self.frame_queues[exam_id].full():
                    self.frame_queues[exam_id].put(frame.copy())
                    
            logger.info(f"Monitoring loop ended for exam {exam_id}")
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            self.stop_flags[exam_id].set()
        finally:
            if exam_id in self.camera_sources:
                self.camera_sources[exam_id].release()
                del self.camera_sources[exam_id]

    def _check_camera_health(self, exam_id: str, cap: cv2.VideoCapture) -> bool:
        """Check and maintain camera health"""
        if not cap.isOpened():
            logger.warning("Camera disconnected, attempting to reconnect...")
            try:
                cap.release()
                new_cap = cv2.VideoCapture(0)
                if new_cap.isOpened():
                    with self.lock:
                        self.camera_sources[exam_id] = new_cap
                    return True
                logger.error("Failed to reconnect to camera")
                return False
            except Exception as e:
                logger.error(f"Camera reconnection failed: {str(e)}")
                return False
        return True

    def _handle_detections(self, exam_id: str, frame: np.ndarray, 
                         detections: List[Detection], settings: Dict) -> None:
        """Handle detected violations"""
        try:
            confidence_threshold = settings.get('confidence_threshold', 0.5)
            
            for detection in detections:
                if detection.confidence < confidence_threshold:
                    continue

                # Create detection record
                detection_record = {
                    "exam_id": exam_id,
                    "timestamp": detection.timestamp.isoformat(),
                    "detection_type": detection.type,
                    "confidence": float(detection.confidence),
                    "details": detection.extra_data or {}
                }
                
                # Insert into database
                self._insert_detection(detection_record)
                
                # Process based on detection type
                self._process_detection_by_type(exam_id, frame, detection, settings)
                    
        except Exception as e:
            logger.error(f"Error handling detections: {str(e)}")

    def _insert_detection(self, detection_record: Dict) -> None:
        """Insert detection record into database"""
        try:
            with self.lock:
                self.db_manager.supabase.table("detections").insert(
                    detection_record
                ).execute()
        except Exception as e:
            logger.error(f"Error inserting detection: {str(e)}")
            raise DatabaseError(f"Failed to insert detection: {str(e)}")

    def _process_detection_by_type(self, exam_id: str, frame: np.ndarray, 
                                 detection: Detection, settings: Dict) -> None:
        """Route detection to appropriate handler"""
        try:
            if detection.type == DetectionType.FACE.value and settings.get('face_detection'):
                self.alert_manager.create_alert(
                    exam_id=exam_id,
                    alert_type=DetectionType.FACE.value,
                    confidence=detection.confidence,
                    frame=frame,
                    timestamp=detection.timestamp
                )
            elif detection.type == DetectionType.OBJECT.value and settings.get('object_detection'):
                self.alert_manager.create_alert(
                    exam_id=exam_id,
                    alert_type=DetectionType.OBJECT.value,
                    confidence=detection.confidence,
                    frame=frame,
                    timestamp=detection.timestamp
                )
            elif detection.type == DetectionType.MOTION.value and settings.get('motion_detection'):
                self.alert_manager.create_alert(
                    exam_id=exam_id,
                    alert_type=DetectionType.MOTION.value,
                    confidence=detection.confidence,
                    frame=frame,
                    timestamp=detection.timestamp
                )
            elif detection.type == DetectionType.PHONE.value and settings.get('object_detection'):
                self.alert_manager.create_alert(
                    exam_id=exam_id,
                    alert_type=DetectionType.PHONE.value,
                    confidence=detection.confidence,
                    frame=frame,
                    timestamp=detection.timestamp
                )
            elif detection.type == DetectionType.GOOD_BEHAVIOR.value:
                # Always record good behavior
                self.alert_manager.create_alert(
                    exam_id=exam_id,
                    alert_type=DetectionType.GOOD_BEHAVIOR.value,
                    confidence=detection.confidence,
                    frame=frame,
                    timestamp=detection.timestamp
                )
        except Exception as e:
            logger.error(f"Error processing detection: {str(e)}")

    def _save_exam_metrics(self, exam_id: str) -> None:
        """Calculate and save exam metrics"""
        try:
            metrics = self.calculate_exam_metrics(exam_id)
            if metrics:
                with self.lock:
                    self.db_manager.supabase.table("exam_metrics").insert(metrics).execute()
                logger.info(f"Metrics saved for exam {exam_id}")
        except Exception as e:
            logger.error(f"Error saving metrics for exam {exam_id}: {str(e)}")

    def calculate_exam_metrics(self, exam_id: str) -> Dict:
        """Generate comprehensive metrics for an exam from detections"""
        try:
            # Get all detections for this exam
            result = self.db_manager.supabase.table("detections") \
                .select("*") \
                .eq("exam_id", exam_id) \
                .execute()

            detections = result.data or []
            total_detections = len(detections)

            if total_detections == 0:
                return {
                    "exam_id": exam_id,
                    "total_detections": 0,
                    "metrics_data": {},
                    "monitoring_duration": None,
                    "created_at": datetime.utcnow().isoformat()
                }

            # Calculate time-based metrics
            timestamps = [datetime.fromisoformat(d["timestamp"]) for d in detections]
            time_range = max(timestamps) - min(timestamps)
            
            # Group detections by type
            type_counts = self._group_by_type(detections)
            
            # Calculate confidence statistics
            confidences = [d["confidence"] for d in detections if d["confidence"] is not None]
            confidence_stats = self._calculate_confidence_stats(confidences)

            # Calculate detection rates
            detection_rate = self._calculate_detection_rate(total_detections, time_range)

            metrics = {
                "exam_id": exam_id,
                "total_detections": total_detections,
                "detection_rate_per_min": detection_rate,
                "monitoring_duration_seconds": time_range.total_seconds(),
                "confidence_stats": confidence_stats,
                "detection_types": type_counts,
                "time_first_detection": min(timestamps).isoformat(),
                "time_last_detection": max(timestamps).isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "metrics_version": "2.0"  # Track metrics format version
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise MonitoringError(f"Failed to calculate metrics: {str(e)}")

    def _calculate_confidence_stats(self, confidences: List[float]) -> Dict:
        """Calculate confidence statistics"""
        if not confidences:
            return {
                "mean": None,
                "median": None,
                "max": None,
                "min": None,
                "std_dev": None,
                "percentiles": {
                    "25": None,
                    "50": None,
                    "75": None,
                    "95": None
                }
            }
            
        return {
            "mean": float(np.mean(confidences)),
            "median": float(np.median(confidences)),
            "max": float(max(confidences)),
            "min": float(min(confidences)),
            "std_dev": float(np.std(confidences)),
            "percentiles": {
                "25": float(np.percentile(confidences, 25)),
                "50": float(np.percentile(confidences, 50)),
                "75": float(np.percentile(confidences, 75)),
                "95": float(np.percentile(confidences, 95))
            }
        }

    def _calculate_detection_rate(self, total_detections: int, 
                                time_range: timedelta) -> float:
        """Calculate detection rate per minute"""
        if time_range.total_seconds() > 0:
            return total_detections / time_range.total_seconds() * 60
        return 0.0

    def _group_by_type(self, detections: List[Dict]) -> Dict[str, Dict]:
        """Group detections by type with additional statistics"""
        type_data = {}
        
        for d in detections:
            dtype = d["detection_type"]
            if dtype not in type_data:
                type_data[dtype] = {
                    "count": 0,
                    "confidences": []
                }
            type_data[dtype]["count"] += 1
            if d["confidence"] is not None:
                type_data[dtype]["confidences"].append(d["confidence"])
        
        # Calculate type-specific stats
        result = {}
        for dtype, data in type_data.items():
            confidences = data["confidences"]
            result[dtype] = {
                "count": data["count"],
                "percentage": (data["count"] / len(detections)) * 100,
                "mean_confidence": float(np.mean(confidences)) if confidences else None,
                "max_confidence": float(max(confidences)) if confidences else None
            }
        
        return result

    # ======================
    # Streamlit UI Helpers
    # ======================
    
    def _get_scheduled_exams(self, now: datetime) -> Any:
        """Get scheduled exams from database"""
        try:
            return self.db_manager.supabase.table("scheduled_exams").select(
                "*"
            ).gte("end_time", now.isoformat()).execute()
        except Exception as e:
            logger.error(f"Error getting scheduled exams: {str(e)}")
            raise DatabaseError(f"Failed to get scheduled exams: {str(e)}")

    def _get_exam_configs(self, scheduled: Any) -> Dict:
        """Get exam configs from database"""
        try:
            config_ids = [exam['exam_id'] for exam in scheduled.data]
            configs = self.db_manager.supabase.table("exams").select(
                "*"
            ).in_("id", config_ids).execute()
            return {c['id']: c for c in configs.data}
        except Exception as e:
            logger.error(f"Error getting exam configs: {str(e)}")
            raise DatabaseError(f"Failed to get exam configs: {str(e)}")

    def _categorize_exams(self, scheduled: Any, config_map: Dict, 
                         now: datetime) -> Tuple[List, List]:
        """Categorize exams into current and upcoming"""
        current_exams = []
        upcoming_exams = []
        
        for exam in scheduled.data:
            config = config_map.get(exam['exam_id'])
            if not config:
                continue
                
            exam_data = {**config, **exam}
            start_time = datetime.fromisoformat(exam['start_time'])
            end_time = datetime.fromisoformat(exam['end_time'])
            
            if start_time <= now <= end_time:
                current_exams.append(exam_data)
            elif start_time > now:
                upcoming_exams.append(exam_data)
        
        return current_exams, upcoming_exams

    def _render_current_exams(self, current_exams: List, now: datetime) -> None:
        """Render current exams section"""
        st.header("🟢 Currently Running Exams")
        if current_exams:
            for exam in current_exams:
                with st.expander(f"📝 {exam['exam_name']} ({exam['course_code']})", expanded=True):
                    self._render_exam_details(exam, now)
        else:
            st.info("No exams currently in progress")

    def _render_exam_details(self, exam: Dict, now: datetime) -> None:
        """Render details for a single exam"""
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Venue:** {exam['venue']}")
            st.write(f"**Instructor:** {exam['instructor']}")
            st.write(f"**Students:** {exam['total_students']}")
        with col2:
            end_time = datetime.fromisoformat(exam['end_time'])
            time_left = end_time - now
            st.write(f"**Time Remaining:** {str(time_left).split('.')[0]}")
            st.write(f"**Type:** {exam['exam_type']}")
            
        # Monitoring controls
        self._render_monitoring_controls(exam)
        
        # Metrics display
        self._render_exam_metrics(exam)

    def _render_monitoring_controls(self, exam: Dict) -> None:
        """Render monitoring controls for an exam"""
        is_monitoring = exam['id'] in self.monitoring_threads
        
        if not is_monitoring:
            if st.button("▶️ Start Monitoring", key=f"start_{exam['id']}"):
                with st.spinner("Starting monitoring..."):
                    self.start_monitoring(exam['id'])
                st.rerun()
        else:
            if st.button("⏹️ Stop Monitoring", key=f"stop_{exam['id']}"):
                with st.spinner("Stopping monitoring and saving metrics..."):
                    self.stop_monitoring(exam['id'])
                st.rerun()
            
            # Show monitoring feed and stats
            self._render_monitoring_feed(exam['id'])
            self._render_monitoring_stats(exam['id'])

    def _render_monitoring_feed(self, exam_id: str) -> None:
        """Render live monitoring feed"""
        st.subheader("📹 Live Feed")
        feed_placeholder = st.empty()
        
        if exam_id in self.frame_queues:
            try:
                frame = self.frame_queues[exam_id].get_nowait()
                feed_placeholder.image(
                    frame,
                    channels="BGR",
                    use_column_width=True,
                    caption="Live Monitoring Feed"
                )
            except queue.Empty:
                pass

    def _render_monitoring_stats(self, exam_id: str) -> None:
        """Render monitoring statistics"""
        stats = self.get_monitoring_statistics(exam_id)
        stat_cols = st.columns(6)
        
        with stat_cols[0]:
            st.metric("Face Detections", stats['face_detections'])
        with stat_cols[1]:
            st.metric("Object Detections", stats['object_detections'])
        with stat_cols[2]:
            st.metric("Motion Detections", stats['motion_detections'])
        with stat_cols[3]:
            st.metric("Person Count", stats['person_detections'])
        with stat_cols[4]:
            st.metric("Phone Detections", stats['phone_detections'])
        with stat_cols[5]:
            st.metric("Good Behavior", stats['good_behavior'])

    def _render_exam_metrics(self, exam: Dict) -> None:
        """Render exam metrics section"""
        if st.button("📊 View Metrics", key=f"metrics_{exam['id']}"):
            try:
                metrics = self.db_manager.supabase.table("exam_metrics") \
                    .select("*") \
                    .eq("exam_id", exam['id']) \
                    .order("created_at", desc=True) \
                    .limit(1) \
                    .execute()

                if metrics.data:
                    self._display_metrics(metrics.data[0], exam['exam_name'])
                else:
                    st.info("No metrics found for this exam yet.")
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
                st.error("Failed to load metrics")

    def _display_metrics(self, metrics: Dict, exam_name: str) -> None:
        """Display metrics in an organized way"""
        st.subheader(f"📊 Metrics for {exam_name}")
        
        # Main metrics
        cols = st.columns(3)
        cols[0].metric("Total Detections", metrics["total_detections"])
        cols[1].metric("Detection Rate", f"{metrics.get('detection_rate_per_min', 0):.1f}/min")
        cols[2].metric("Monitoring Duration", 
                      f"{metrics.get('monitoring_duration_seconds', 0)/60:.1f} mins")
        
        # Confidence stats
        st.subheader("Confidence Statistics")
        if metrics.get("confidence_stats"):
            conf_stats = metrics["confidence_stats"]
            conf_cols = st.columns(4)
            conf_cols[0].metric("Mean", f"{conf_stats.get('mean', 0):.2f}")
            conf_cols[1].metric("Median", f"{conf_stats.get('median', 0):.2f}")
            conf_cols[2].metric("Std Dev", f"{conf_stats.get('std_dev', 0):.2f}")
            conf_cols[3].metric("Range", 
                               f"{conf_stats.get('min', 0):.2f}-{conf_stats.get('max', 0):.2f}")
        
        # Detection type breakdown
        st.subheader("Detection Type Breakdown")
        if metrics.get("detection_types"):
            type_data = []
            for dtype, stats in metrics["detection_types"].items():
                type_data.append({
                    "Type": dtype,
                    "Count": stats["count"],
                    "Percentage": f"{stats.get('percentage', 0):.1f}%",
                    "Mean Confidence": f"{stats.get('mean_confidence', 0):.2f}" 
                                      if stats.get("mean_confidence") is not None else "N/A"
                })
            
            df_types = pd.DataFrame(type_data)
            st.dataframe(df_types, use_container_width=True, hide_index=True)

    def _render_upcoming_exams(self, upcoming_exams: List) -> None:
        """Render upcoming exams section"""
        st.header("📅 Upcoming Exams")
        if upcoming_exams:
            df = pd.DataFrame(upcoming_exams)
            
            # Format datetime columns
            if 'start_time' in df.columns:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['start_time'] = df['start_time'].dt.strftime('%d/%m/%Y %H:%M')
            
            st.dataframe(
                df,
                column_config={
                    "exam_name": "Exam Name",
                    "course_code": "Course",
                    "start_time": "Start Time",
                    "venue": "Venue",
                    "instructor": "Instructor",
                    "total_students": "Students"
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No upcoming exams scheduled")
