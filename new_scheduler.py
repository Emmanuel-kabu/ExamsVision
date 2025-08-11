import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import pytz

logger = logging.getLogger(__name__)

class ExamScheduler:
    """Handles exam scheduling and calendar display"""
    
    def __init__(self, db_manager):
        if not db_manager:
            raise ValueError("Database manager is required for ExamScheduler")
        if not hasattr(db_manager, 'supabase') or not db_manager.supabase:
            raise ValueError("Database manager must have an active Supabase client")
            
        self.db_manager = db_manager
        
        # Test database connection on initialization
        try:
            test_response = self.db_manager.supabase.table('exams').select('count').limit(1).execute()
            if not test_response:
                raise Exception("Could not verify database access")
            logger.info("ExamScheduler initialized successfully with database connection")
        except Exception as e:
            logger.error(f"Failed to initialize ExamScheduler: {str(e)}")
            raise
        
    def render_scheduler(self):
        """Render the exam scheduling interface"""
        st.title("📅 Exam Scheduler")
        
        try:
            logger.info("Loading exam scheduler...")
            # Test database connection
            try:
                test_query = self.db_manager.supabase.table("exams").select("id").limit(1).execute()
                logger.info("Database connection successful")
            except Exception as db_error:
                logger.error(f"Database connection failed: {str(db_error)}")
                raise Exception("Could not connect to database. Please check your connection.")

            # Get all exams from the exams table
            exams_response = self.db_manager.supabase.table("exams").select("*").execute()
            if not exams_response or not hasattr(exams_response, 'data'):
                logger.error("Failed to retrieve exams from database")
                raise Exception("Could not retrieve exam data from database")

            # Split exams based on status
            configured_exams = [exam for exam in exams_response.data if exam.get('status') == 'configured']
            scheduled_exams = [exam for exam in exams_response.data if exam.get('status') in ['scheduled', 'running', 'completed']]

            logger.info(f"Retrieved {len(configured_exams)} configured and {len(scheduled_exams)} scheduled exams")

            # All exams
            all_exams = exams_response.data if hasattr(exams_response, 'data') else exams_response.get('data', [])
            if not all_exams:
                logger.warning("No exams found in database")

            # Categorize
            now = datetime.now(pytz.UTC)
            past_exams, current_exams, upcoming_exams, available_exams = [], [], [], []

            def parse_exam_time(time_str):
                try:
                    return datetime.fromisoformat(time_str) if time_str else None
                except ValueError:
                    return None

            def categorize_exam(exam):
                required_fields = ['start_time', 'end_time', 'exam_name', 'course_code', 'status']
                if any(not exam.get(f) for f in required_fields):
                    return None
                start_time = parse_exam_time(exam['start_time'])
                end_time = parse_exam_time(exam['end_time'])
                if not start_time or not end_time:
                    return None
                if exam['status'].lower() == 'configured':
                    return 'available'
                elif end_time < now:
                    return 'past'
                elif start_time <= now <= end_time:
                    return 'current'
                elif start_time > now:
                    return 'upcoming'
                else:
                    return 'past'

            for exam in all_exams:
                category = categorize_exam(exam)
                if category == 'past': past_exams.append(exam)
                elif category == 'current': current_exams.append(exam)
                elif category == 'upcoming': upcoming_exams.append(exam)
                elif category == 'available': available_exams.append(exam)

            # Sort
            past_exams.sort(key=lambda x: x['start_time'], reverse=True)
            upcoming_exams.sort(key=lambda x: x['start_time'])

            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📅 Calendar View", "📋 Schedule New", "🔄 Current Exams", "📆 Upcoming Exams", "📚 Past Exams"
            ])
            
            with tab2:
                st.subheader("📋 Schedule New Exam")
                pending_exam_id = st.session_state.get('pending_exam_id')
                pending_exam = next((exam for exam in available_exams if exam['id'] == pending_exam_id), None) if pending_exam_id else None

                if pending_exam or available_exams:
                    with st.form("schedule_exam"):
                        if pending_exam:
                            selected_exam = pending_exam
                            st.info(f"Scheduling exam: {selected_exam['exam_name']} ({selected_exam['course_code']})")
                        else:
                            selected_exam = st.selectbox(
                                "Select Exam to Schedule",
                                options=available_exams,
                                format_func=lambda x: f"{x['exam_name']} ({x['course_code']})"
                            )
                        col1, col2 = st.columns(2)
                        with col1:
                            date = st.date_input("Exam Date", min_value=datetime.now().date())
                        with col2:
                            time = st.time_input("Start Time", value=datetime.now().replace(hour=9, minute=0))
                        
                        if st.form_submit_button("📅 Schedule Exam", use_container_width=True):
                            try:
                                start_datetime = datetime.combine(date, time)
                                end_datetime = start_datetime + timedelta(minutes=selected_exam['duration'])

                                if self._check_schedule_conflicts(start_datetime, end_datetime, selected_exam['venue']):
                                    st.error("Schedule conflict detected! Another exam is scheduled for this time and venue.")
                                    return
                                
                                # 1️⃣ Update exams table
                                update_data = {
                                    'exam_name': selected_exam['exam_name'],
                                    'course_code': selected_exam['course_code'],
                                    'department': selected_exam['department'],
                                    'instructor': selected_exam['instructor'],
                                    'total_students': selected_exam['total_students'],
                                    'venue': selected_exam['venue'],
                                    'duration': selected_exam['duration'],
                                    'exam_type': selected_exam['exam_type'],
                                    'status': 'scheduled',
                                    'start_time': start_datetime.isoformat(),
                                    'end_time': end_datetime.isoformat()
                                }
                                self.db_manager.update_exam(selected_exam['id'], update_data)

                                # 2️⃣ Insert into scheduled_exams table
                                self.db_manager.supabase.table("scheduled_exams").insert({
                                    'exam_id': selected_exam['id'],
                                    'exam_name': selected_exam['exam_name'],
                                    'course_code': selected_exam['course_code'],
                                    'start_time': start_datetime.isoformat(),
                                    'end_time': end_datetime.isoformat(),
                                    'venue': selected_exam['venue'],
                                    'instructor': selected_exam['instructor'],
                                    'total_students': selected_exam['total_students'],
                                    'status': 'scheduled'
                                }).execute()

                                st.success(f"Exam {selected_exam['exam_name']} scheduled successfully!")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Error scheduling exam: {e}")
                                st.error(f"Failed to schedule exam: {str(e)}")
                else:
                    st.info("No configured exams available.")
                    
        except Exception as e:
            logger.error(f"Error rendering scheduler: {str(e)}")
            st.error(f"Failed to load exam scheduler: {str(e)}")

    def _check_schedule_conflicts(self, start_time: datetime, end_time: datetime, venue: str) -> bool:
        """Check for scheduling conflicts"""
        try:
            venue_exams = self.db_manager.supabase.table("exams").select("id,start_time,end_time").eq("venue", venue).eq("status", "scheduled").execute()
            if not venue_exams.data:
                return False
            for exam in venue_exams.data:
                exam_start = datetime.fromisoformat(exam['start_time'].replace('Z', '+00:00'))
                exam_end = datetime.fromisoformat(exam['end_time'].replace('Z', '+00:00'))
                if (start_time < exam_end and end_time > exam_start):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking schedule conflicts: {e}")
            return True

