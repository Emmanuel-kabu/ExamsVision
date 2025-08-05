import streamlit as st
from datetime import datetime, timedelta
import logging
import pytz
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class LiveMonitoring:
    """Handles live exam monitoring and upcoming exam display"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def get_active_and_upcoming_exams(self):
        """Fetch active and upcoming exams from the database"""
        try:
            # Get current time in UTC
            now = datetime.now(pytz.UTC)
            
            # Get scheduled exams that are not in the past
            scheduled_exams = self.db_manager.supabase.table("scheduled_exams").select(
                "id",
                "exam_config_id",
                "start_time",
                "end_time",
                "venue",
                "status"
            ).gte("end_time", now.isoformat()).execute()
            
            if not scheduled_exams.data:
                return [], []
                
            # Get corresponding exam configs
            config_ids = [exam['exam_config_id'] for exam in scheduled_exams.data]
            configs = self.db_manager.supabase.table("exam_configs").select(
                "id",
                "exam_name",
                "exam_type",
                "course_code",
                "department",
                "instructor",
                "total_students",
                "venue",
                "duration"
            ).in_("id", config_ids).execute()
            
            # Create a mapping of config_id to config data
            config_map = {config['id']: config for config in configs.data}
            
            # Combine and categorize exams
            current_exams = []
            upcoming_exams = []
            
            for scheduled in scheduled_exams.data:
                config = config_map.get(scheduled['exam_config_id'])
                if config:
                    exam_data = {**config, **scheduled}
                    start_time = datetime.fromisoformat(scheduled['start_time'])
                    end_time = datetime.fromisoformat(scheduled['end_time'])
                    
                    if start_time <= now <= end_time:
                        current_exams.append(exam_data)
                    elif start_time > now:
                        upcoming_exams.append(exam_data)
            
            # Sort by start time
            current_exams.sort(key=lambda x: x['start_time'])
            upcoming_exams.sort(key=lambda x: x['start_time'])
            
            return current_exams, upcoming_exams
            
        except Exception as e:
            logger.error(f"Error fetching active and upcoming exams: {e}")
            return [], []
    
    def render_monitoring_interface(self):
        """Render the live monitoring interface"""
        st.title("🔍 Live Exam Monitoring")
        
        try:
            current_exams, upcoming_exams = self.get_active_and_upcoming_exams()
            
            # Display current exams
            st.header("🟢 Currently Running Exams")
            if current_exams:
                for exam in current_exams:
                    with st.expander(f"📝 {exam['exam_name']} - {exam['course_code']}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Instructor:** {exam['instructor']}")
                            st.write(f"**Department:** {exam['department']}")
                            st.write(f"**Venue:** {exam['venue']}")
                        with col2:
                            st.write(f"**Type:** {exam['exam_type']}")
                            st.write(f"**Students:** {exam['total_students']}")
                            end_time = datetime.fromisoformat(exam['end_time'])
                            time_left = end_time - datetime.now(pytz.UTC)
                            st.write(f"**Time Left:** {str(time_left).split('.')[0]}")
                        
                        # Add monitoring controls
                        st.button("▶️ Start Monitoring", key=f"monitor_{exam['id']}")
                        
                        # Show monitoring metrics (placeholder for now)
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Face Detections", "0")
                        with metrics_col2:
                            st.metric("Violations", "0")
                        with metrics_col3:
                            st.metric("Alerts", "0")
            else:
                st.info("No exams are currently running")
            
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
