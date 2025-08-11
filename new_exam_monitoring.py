import streamlit as st
import cv2
import logging
import threading
import queue
import numpy as np
from datetime import datetime
import pandas as pd
import os
import pytz
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class ExamMonitoringSystem:
    """Handles real-time exam monitoring and alerts"""
    
    def __init__(self, db_manager, video_processor, alert_manager):
        self.db_manager = db_manager
        self.video_processor = video_processor
        self.alert_manager = alert_manager
        self.monitoring_threads = {}
        self.frame_queues = {}
        self.stop_flags = {}
        
    def get_active_exam(self, exam_id: str) -> Optional[Dict]:
        """Get active exam details from database"""
        try:
            # Get scheduled exam
            scheduled = self.db_manager.supabase.table("scheduled_exams").select(
                "*"
            ).eq("id", exam_id).single().execute()
            
            if not scheduled.data:
                return None
            
            # Get exam config
            config = self.db_manager.supabase.table("exam_configs").select(
                "*"
            ).eq("id", scheduled.data['exam_config_id']).single().execute()
            
            if not config.data:
                return None
                
            return {**config.data, **scheduled.data}
            
        except Exception as e:
            logger.error(f"Error getting active exam: {e}")
            return None
    
    def start_monitoring(self, exam_id: str):
        """Start monitoring for a specific exam"""
        if exam_id in self.monitoring_threads:
            logger.warning(f"Monitoring already active for exam {exam_id}")
            return
        
        exam_data = self.get_active_exam(exam_id)
        if not exam_data:
            logger.error(f"Could not find exam with ID {exam_id}")
            return
            
        # Initialize frame queue and stop flag
        self.frame_queues[exam_id] = queue.Queue(maxsize=10)
        self.stop_flags[exam_id] = threading.Event()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(exam_id, exam_data),
            daemon=True
        )
        self.monitoring_threads[exam_id] = monitor_thread
        monitor_thread.start()
        
        logger.info(f"Started monitoring for exam {exam_data['exam_name']}")
        
    def stop_monitoring(self, exam_id: str):
        """Stop monitoring for a specific exam"""
        if exam_id in self.stop_flags:
            self.stop_flags[exam_id].set()
            if exam_id in self.monitoring_threads:
                self.monitoring_threads[exam_id].join()
                del self.monitoring_threads[exam_id]
            del self.stop_flags[exam_id]
            del self.frame_queues[exam_id]
            logger.info(f"Stopped monitoring for exam {exam_id}")
    
    def _monitoring_loop(self, exam_id: str, exam_data: Dict):
        """Main monitoring loop for an exam"""
        try:
            cap = cv2.VideoCapture(0)  # Can be changed to other camera sources
            if not cap.isOpened():
                logger.error("Failed to open camera")
                return
                
            while not self.stop_flags[exam_id].is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame for violations
                detections = self.video_processor.process_frame(frame)
                
                # Check monitoring settings from exam configuration
                settings = exam_data.get('monitoring_settings', {})
                
                # Process detections based on settings
                if detections:
                    self._handle_detections(exam_id, frame, detections, settings)
                
                # Update frame queue for display
                if not self.frame_queues[exam_id].full():
                    self.frame_queues[exam_id].put(frame)
                    
            cap.release()
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.stop_flags[exam_id].set()
    
    def _handle_detections(self, exam_id: str, frame, detections, settings):
        """Handle detected violations"""
        try:
            confidence_threshold = settings.get('confidence_threshold', 0.5)
            
            for detection in detections:
                if detection.confidence < confidence_threshold:
                    continue
                    
                # Process different types of detections based on settings
                if detection.is_face and settings.get('face_detection'):
                    self._handle_face_detection(exam_id, frame, detection)
                elif detection.is_object and settings.get('object_detection'):
                    self._handle_object_detection(exam_id, frame, detection)
                elif detection.is_motion and settings.get('motion_detection'):
                    self._handle_motion_detection(exam_id, frame, detection)
                    
        except Exception as e:
            logger.error(f"Error handling detections: {e}")
    
    def _handle_face_detection(self, exam_id, frame, detection):
        """Handle face detection alerts"""
        self.alert_manager.create_alert(
            exam_id=exam_id,
            alert_type="face_detection",
            confidence=detection.confidence,
            frame=frame
        )
    
    def _handle_object_detection(self, exam_id, frame, detection):
        """Handle object detection alerts"""
        self.alert_manager.create_alert(
            exam_id=exam_id,
            alert_type="object_detection",
            confidence=detection.confidence,
            frame=frame
        )
    
    def _handle_motion_detection(self, exam_id, frame, detection):
        """Handle motion detection alerts"""
        self.alert_manager.create_alert(
            exam_id=exam_id,
            alert_type="motion_detection",
            confidence=detection.confidence,
            frame=frame
        )
    
    def get_monitoring_statistics(self, exam_id: str) -> Dict:
        """Get current monitoring statistics for an exam"""
        try:
            stats = self.alert_manager.get_exam_statistics(exam_id)
            return {
                "face_detections": stats.get("face_detection", 0),
                "object_detections": stats.get("object_detection", 0),
                "motion_detections": stats.get("motion_detection", 0),
                "total_alerts": sum(stats.values())
            }
        except Exception as e:
            logger.error(f"Error getting monitoring statistics: {e}")
            return {
                "face_detections": 0,
                "object_detections": 0,
                "motion_detections": 0,
                "total_alerts": 0
            }
    
    def render_monitoring_interface(self):
        """Render the monitoring interface in Streamlit"""
        st.title("🔍 Live Exam Monitoring")
        
        try:
            # Get current and upcoming exams
            now = datetime.now(pytz.UTC)
            
            # Get scheduled exams that are happening now or in the future
            scheduled = self.db_manager.supabase.table("scheduled_exams").select(
                "*"
            ).gte("end_time", now.isoformat()).execute()
            
            if not scheduled.data:
                st.info("No active or upcoming exams found")
                return
                
            # Get corresponding exam configs
            config_ids = [exam['exam_config_id'] for exam in scheduled.data]
            configs = self.db_manager.supabase.table("exam_configs").select(
                "*"
            ).in_("id", config_ids).execute()
            
            config_map = {c['id']: c for c in configs.data}
            
            # Separate current and upcoming exams
            current_exams = []
            upcoming_exams = []
            
            for exam in scheduled.data:
                config = config_map.get(exam['exam_config_id'])
                if not config:
                    continue
                    
                exam_data = {**config, **exam}
                start_time = datetime.fromisoformat(exam['start_time'])
                end_time = datetime.fromisoformat(exam['end_time'])
                
                if start_time <= now <= end_time:
                    current_exams.append(exam_data)
                elif start_time > now:
                    upcoming_exams.append(exam_data)
            
            # Display current exams
            st.header("🟢 Currently Running Exams")
            if current_exams:
                for exam in current_exams:
                    with st.expander(f"📝 {exam['exam_name']} ({exam['course_code']})", expanded=True):
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
                        is_monitoring = exam['id'] in self.monitoring_threads
                        if not is_monitoring:
                            if st.button("▶️ Start Monitoring", key=f"start_{exam['id']}"):
                                self.start_monitoring(exam['id'])
                                st.rerun()
                        else:
                            if st.button("⏹️ Stop Monitoring", key=f"stop_{exam['id']}"):
                                self.stop_monitoring(exam['id'])
                                st.rerun()
                        
                        if is_monitoring:
                            # Show monitoring feed
                            st.subheader("📹 Live Feed")
                            feed_placeholder = st.empty()
                            
                            # Show statistics
                            stats = self.get_monitoring_statistics(exam['id'])
                            stat_cols = st.columns(4)
                            with stat_cols[0]:
                                st.metric("Face Detections", stats['face_detections'])
                            with stat_cols[1]:
                                st.metric("Object Detections", stats['object_detections'])
                            with stat_cols[2]:
                                st.metric("Motion Detections", stats['motion_detections'])
                            with stat_cols[3]:
                                st.metric("Total Alerts", stats['total_alerts'])
                            
                            # Update live feed
                            if exam['id'] in self.frame_queues:
                                try:
                                    frame = self.frame_queues[exam['id']].get_nowait()
                                    feed_placeholder.image(
                                        frame,
                                        channels="BGR",
                                        use_column_width=True,
                                        caption="Live Monitoring Feed"
                                    )
                                except queue.Empty:
                                    pass
            else:
                st.info("No exams currently in progress")
            
            # Display upcoming exams
            st.header("📅 Upcoming Exams")
            if upcoming_exams:
                df = pd.DataFrame(upcoming_exams)
                st.dataframe(
                    df,
                    column_config={
                        "exam_name": "Exam Name",
                        "course_code": "Course",
                        "start_time": st.column_config.DatetimeColumn(
                            "Start Time",
                            format="DD/MM/YYYY HH:mm"
                        ),
                        "venue": "Venue",
                        "instructor": "Instructor",
                        "total_students": "Students"
                    },
                    use_container_width=True
                )
            else:
                st.info("No upcoming exams scheduled")
                
        except Exception as e:
            logger.error(f"Error in monitoring interface: {e}")
            st.error("Failed to load monitoring interface")