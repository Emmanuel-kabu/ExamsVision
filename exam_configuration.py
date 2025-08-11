import streamlit as st
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
from database_manager import DatabaseManager, init_supabase
import pandas as pd

logger = logging.getLogger(__name__)

class ExamConfiguration:
    """Handles exam configuration and scheduling"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def render_exam_form(self):
        """Render the exam configuration form"""
        st.title("🛠️ Configure Exam")
        
        with st.form("exam_configuration"):
            # Basic Information
            st.subheader("📝 Basic Information")
            col1, col2 = st.columns(2)
            with col1:
                exam_name = st.text_input("Exam Name*", help="Enter a descriptive name for the exam")
                course_code = st.text_input("Course Code*", help="Enter the course code (e.g., CS101)")
                instructor = st.text_input("Instructor Name*")
            with col2:
                department = st.text_input("Department*")
                degree_type = st.selectbox("Degree Type*", ["BSc", "MSc", "PhD", "Other"])
                year_of_study = st.selectbox("Year of Study*", ["1st Year", "2nd Year", "3rd Year", "4th Year", "5th Year", "6th Year", "7th Year", "8th Year", "Other"])
                exam_type = st.selectbox("Exam Type*", [
                    "Quiz",
                    "Mid Semester",
                    "End of Semester",
                    "Lab Test",
                    "Project Defense",
                    "Assignment",
                    "Other"
                ])
            
            # Scheduling
            st.subheader("📅 Scheduling")
            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input("Start Date*", min_value=datetime.now().date())
                start_time = st.time_input("Start Time*", value=datetime.now().replace(hour=9, minute=0))
                venue = st.text_input("Venue*", help="Enter the exam location")
            with col4:
                duration = st.number_input("Duration (minutes)*", min_value=30, value=120)
                total_students = st.number_input("Total Students*", min_value=1, value=30)
            
            # Monitoring Settings
            st.subheader("⚙️ Monitoring Settings")
            col5, col6 = st.columns(2)
            with col5:
                face_detection = st.checkbox("Enable Face Detection", value=True)
                noise_detection = st.checkbox("Enable Noise Detection")
                object_detection = st.checkbox("Enable Object Detection")
            with col6:
                multi_face_detection = st.checkbox("Enable Multi-Face Detection", value=True)
                motion_detection = st.checkbox("Enable Motion Detection", value=True)
            
            confidence_threshold = st.slider(
                "Detection Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Set the minimum confidence level for detections"
            )
            
            st.markdown("---")
            submit_button = st.form_submit_button("💾 Save Exam Configuration")
            
            if submit_button:
                try:
                    # Debug exam type
                    logger.info(f"Selected exam_type: {exam_type}")
                    if not exam_type:
                        st.error("Exam Type is required")
                        return
                    
                    # Construct start and end datetimes
                    start_datetime = datetime.combine(start_date, start_time)
                    end_datetime = start_datetime + timedelta(minutes=duration)
                    
                    # Prepare exam data with explicit exam type validation
                    if not isinstance(exam_type, str) or not exam_type.strip():
                        st.error("Invalid exam type selected")
                        return

                    # Use the selected date and time for initial values
                    initial_start = datetime.combine(start_date, start_time)
                    initial_end = initial_start + timedelta(minutes=duration)
                    
                    exam_data = {
                        "exam_name": exam_name,
                        "exam_type": exam_type.strip(),  # Ensure clean string value
                        "course_code": course_code,
                        "department": department,
                        "instructor": instructor,
                        "degree_type": degree_type,
                        "year_of_study": year_of_study,
                        "total_students": total_students,
                        "start_time": initial_start.isoformat(),  # Use initial date/time
                        "end_time": initial_end.isoformat(),      # Calculate end time
                        "duration": duration,
                        "status": "configured",  # Start as configured, not scheduled
                        "venue": venue
                    }
                    
                    # Store monitoring settings in session state for later use
                    st.session_state.monitoring_settings = {
                        "face_detection": face_detection,
                        "noise_detection": noise_detection,
                        "object_detection": object_detection,
                        "multi_face_detection": multi_face_detection,
                        "motion_detection": motion_detection,
                        "confidence_threshold": confidence_threshold
                    }
                    
                    # Validate required fields
                    required_fields = ["exam_name", "course_code", "instructor", "department", "venue", "exam_type"]
                    missing_fields = [field for field in required_fields if not exam_data.get(field)]
                    
                    if missing_fields:
                        st.error(f"Please fill in all required fields: {', '.join(missing_fields)}")
                        return
                    
                    # Ensure only valid columns are inserted
                    valid_columns = {
                        'exam_name', 'course_code', 'department', 'instructor',
                        'total_students', 'venue', 'duration', 'status',
                        'start_time', 'end_time', 'exam_type',
                        'degree_type', 'year_of_study'
                    }
                    
                    # Filter exam_data to only include valid columns
                    filtered_exam_data = {k: v for k, v in exam_data.items() if k in valid_columns}

                    # Save to exams table
                    result = self.db_manager.supabase.table("exams").insert(filtered_exam_data).execute()
                    
                    if result.data:
                        exam_id = result.data[0]['id']  # Get the new exam config ID
                        st.success("✅ Exam configuration saved successfully!")
                        # Store exam ID in session state for scheduling
                        st.session_state.pending_exam_id = exam_id
                        # Redirect to scheduling page
                        st.session_state.current_page = "Exam Scheduling"
                        st.rerun()
                    else:
                        st.error("Failed to save exam configuration")
                        
                except Exception as e:
                    logger.error(f"Error saving exam configuration: {e}")
                    st.error(f"An error occurred: {str(e)}")
                    
        # Show configuration preview
        if st.checkbox("Show Previous Configuration"):
            try:
                # Get all exam configs
                configs = self.db_manager.supabase.table("exams").select(
                    "id",
                    "exam_name",
                    "exam_type",
                    "course_code",
                    "department",
                    "instructor",
                    "degree_type",
                    "year_of_study",
                    "total_students",
                    "venue",
                    "duration",
                    "status"
                ).execute()
                
                # Get scheduled exams
                scheduled = self.db_manager.supabase.table("scheduled_exams").select(
                    "exam_config_id",
                    "start_time",
                    "end_time",
                    "status"
                ).execute()
                
                if configs.data:
                    # Create a mapping of config_id to scheduled exam data
                    scheduled_map = {s['exam_id']: s for s in scheduled.data}
                    
                    # Combine the data
                    combined_data = []
                    for config in configs.data:
                        exam_data = config.copy()
                        if config['id'] in scheduled_map:
                            exam_data.update({
                                'start_time': scheduled_map[config['id']]['start_time'],
                                'end_time': scheduled_map[config['id']]['end_time'],
                                'status': scheduled_map[config['id']]['status']
                            })
                        combined_data.append(exam_data)
                    # Create DataFrame from combined data
                    df = pd.DataFrame(combined_data)
                    
                    # Sort by start time if available
                    if 'start_time' in df.columns:
                        df = df.sort_values('start_time', ascending=True)
                    
                    st.dataframe(
                        df,
                        column_config={
                            "exam_name": "Exam Name",
                            "exam_type": "Type",
                            "course_code": "Course",
                            "department": "Department",
                            "degree_type": "Degree",
                            "year_of_study": "Year",
                            "start_time": st.column_config.DatetimeColumn(
                                "Start Time",
                                format="DD/MM/YYYY HH:mm",
                                help="Scheduled start time"
                            ),
                            "duration": "Duration (min)",
                            "status": st.column_config.TextColumn(
                                "Status",
                                help="Current status of the exam"
                            ),
                            "venue": "Venue"
                        },
                        use_container_width=True
                    )
            except Exception as e:
                logger.error(f"Error loading exam configurations: {e}")
                st.error("Failed to load existing configurations")