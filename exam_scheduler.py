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
            # First, test the database connection
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
            configured_exams = [
                exam for exam in exams_response.data 
                if exam.get('status') == 'configured'
            ]

            scheduled_exams = [
                exam for exam in exams_response.data 
                if exam.get('status') in ['scheduled', 'running', 'completed']
            ]

            logger.info(f"Retrieved {len(configured_exams)} configured and {len(scheduled_exams)} scheduled exams")

            # Get all exams from response
            all_exams = []
            if hasattr(exams_response, 'data'):
                all_exams = exams_response.data
            elif isinstance(exams_response, dict):
                all_exams = exams_response.get('data', [])
            
            if not all_exams:
                logger.warning("No exams found in database")
                all_exams = []
            
            # Initialize exam categories
            now = datetime.now(pytz.UTC)
            past_exams = []
            current_exams = []
            upcoming_exams = []
            available_exams = []
            
            def parse_exam_time(time_str):
                """Parse time string to datetime object"""
                try:
                    return datetime.fromisoformat(time_str) if time_str else None
                except ValueError as e:
                    return None
            
            def categorize_exam(exam):
                """Categorize a single exam"""
                try:
                    # Check for required fields
                    required_fields = ['start_time', 'end_time', 'exam_name', 'course_code', 'status']
                    missing_fields = [field for field in required_fields if not exam.get(field)]
                    if missing_fields:
                        logger.warning(f"Exam {exam.get('id', 'unknown')} missing required fields: {missing_fields}")
                        return None
                    
                    # Parse times first
                    start_time = parse_exam_time(exam['start_time'])
                    end_time = parse_exam_time(exam['end_time'])
                    
                    if not start_time or not end_time:
                        logger.warning(f"Invalid datetime for exam {exam.get('id', 'unknown')}")
                        return None
                    
                    # Determine category based on time and status
                    if exam['status'].lower() == 'configured':
                        return 'available'
                    elif end_time < now:
                        return 'past'
                    elif start_time <= now and now <= end_time:
                        return 'current'
                    elif start_time > now:
                        return 'upcoming'
                    else:
                        return 'past'
                        
                except Exception as e:
                    logger.error(f"Error categorizing exam {exam.get('id', 'unknown')}: {str(e)}")
                    return None
            
            # Process all exams
            for exam in all_exams:
                try:
                    category = categorize_exam(exam)
                    if category == 'past':
                        past_exams.append(exam)
                    elif category == 'current':
                        current_exams.append(exam)
                    elif category == 'upcoming':
                        upcoming_exams.append(exam)
                    elif category == 'available':
                        available_exams.append(exam)
                except Exception as e:
                    logger.error(f"Error categorizing exam {exam.get('id', 'unknown')}: {str(e)}")
                    continue

            logger.info(f"Categorized exams: {len(past_exams)} past, {len(current_exams)} current, {len(upcoming_exams)} upcoming, {len(available_exams)} available")
            
            # Sort exams by date
            past_exams.sort(key=lambda x: x['start_time'], reverse=True)
            upcoming_exams.sort(key=lambda x: x['start_time'])
            
            # Initialize timezone-aware now
            now = datetime.now(pytz.UTC)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📅 Calendar View", 
                "📋 Schedule New", 
                "🔄 Current Exams", 
                "📆 Upcoming Exams",
                "📚 Past Exams"
            ])
            
            with tab1:
                st.subheader("📆 Calendar View")
                
                # Get current week dates
                today = now.date()
                week_start = today - timedelta(days=today.weekday())
                dates = [week_start + timedelta(days=i) for i in range(7)]
                
                # Create calendar columns
                cols = st.columns(7)
                for i, date in enumerate(dates):
                    with cols[i]:
                        # Format date header
                        if date == today:
                            st.markdown(f"**{date.strftime('%a')}\n\n{date.strftime('%d')}**")
                        else:
                            st.write(f"{date.strftime('%a')}\n\n{date.strftime('%d')}")
                        
                        # Display scheduled exams for this date
                        for exam in upcoming_exams + current_exams:
                            try:
                                exam_date = datetime.fromisoformat(exam['start_time'].replace('Z', '+00:00')).date()
                                if exam_date == date:
                                    with st.expander(f"{datetime.fromisoformat(exam['start_time'].replace('Z', '+00:00')).strftime('%H:%M')} - {exam['course_code']}", expanded=True):
                                        st.write(f"**{exam['exam_name']}**")
                                        st.write(f"📍 {exam['venue']}")
                                        st.write(f"👨‍🏫 {exam['instructor']}")
                            except Exception as e:
                                logger.error(f"Error displaying exam in calendar: {e}")
                                continue
                
                st.markdown("---")
                if st.button("➕ Configure New Exam", key="cal_new_exam", use_container_width=True):
                    st.session_state.current_page = "Exam Configuration"
                    st.rerun()
                    
            with tab2:
                st.subheader("📋 Schedule New Exam")
                
                # Get the pending exam from session state if it exists
                pending_exam_id = st.session_state.get('pending_exam_id')
                pending_exam = None
                
                if pending_exam_id:
                    # Get the exam configuration
                    pending_exam = next((exam for exam in available_exams if exam['id'] == pending_exam_id), None)
                
                if pending_exam or available_exams:
                    # Create scheduling form
                    with st.form("schedule_exam"):
                        if pending_exam:
                            selected_exam = pending_exam
                            st.info(f"Scheduling exam: {selected_exam['exam_name']} ({selected_exam['course_code']})")
                        else:
                            # Select exam
                            if available_exams:
                                selected_exam = st.selectbox(
                                    "Select Exam to Schedule",
                                    options=available_exams,
                                    format_func=lambda x: f"{x['exam_name']} ({x['course_code']})"
                                )
                            else:
                                st.warning("No exams available for scheduling")
                                return
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            date = st.date_input(
                                "Exam Date",
                                min_value=datetime.now().date(),
                                help="Select the date for the exam"
                            )
                        with col2:
                            time = st.time_input(
                                "Start Time",
                                value=datetime.now().replace(hour=9, minute=0),
                                help="Select the start time for the exam"
                            )
                        
                        # Submit button
                        if st.form_submit_button("📅 Schedule Exam", use_container_width=True):
                            try:
                                # Combine date and time
                                start_datetime = datetime.combine(date, time)
                                end_datetime = start_datetime + timedelta(minutes=selected_exam['duration'])

                                # Check for schedule conflicts
                                schedule_conflicts = self._check_schedule_conflicts(
                                    start_datetime,
                                    end_datetime,
                                    selected_exam['venue']
                                )

                                if schedule_conflicts:
                                    st.error("Schedule conflict detected! Another exam is scheduled for this time and venue.")
                                    return

                                # Prepare update data with all required fields
                                update_data = {
                                    'exam_name': selected_exam['exam_name'],
                                    'course_code': selected_exam['course_code'],
                                    'department': selected_exam['department'],
                                    'instructor': selected_exam['instructor'],
                                    'total_students': selected_exam['total_students'],
                                    'venue': selected_exam['venue'],
                                    'duration': selected_exam['duration'],
                                    'exam_type': selected_exam['exam_type'],  # This must exist from configuration
                                    'status': 'scheduled',
                                    'start_time': start_datetime.isoformat(),
                                    'end_time': end_datetime.isoformat()
                                }

                                self.db_manager.update_exam(
                                    selected_exam['id'],
                                    update_data
                                )

                                # Insert into scheduled_exams table
                                schedule_data = {
                                    'exam_id': selected_exam['id'],
                                    'exam_name': selected_exam['exam_name'],
                                    'course_code': selected_exam['course_code'],
                                    'start_time': start_datetime.isoformat(),
                                    'end_time': end_datetime.isoformat(),
                                    'venue': selected_exam['venue'],
                                    'instructor': selected_exam['instructor'],
                                    'total_students': selected_exam['total_students'],
                                    'status': 'scheduled'
                                }
                                insert_result = self.db_manager.supabase.table("scheduled_exams").insert(schedule_data).execute()

                                if hasattr(insert_result, 'error') and insert_result.error:
                                    st.error(f"Failed to save to scheduled_exams: {insert_result.error}")
                                else:
                                    st.success(f"Exam {selected_exam['exam_name']} scheduled successfully!")
                                    st.rerun()
                            except Exception as e:
                                logger.error(f"Error scheduling exam: {e}")
                                st.error(f"Failed to schedule exam: {str(e)}")
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info("No configured exams available. Go to Exam Configuration to configure new exams first.")
                    with col2:
                        if st.button("➕ Configure New Exam", key="schedule_new_exam", use_container_width=True):
                            st.session_state.current_page = "Exam Configuration"
                            st.rerun()
            
            # Display current, upcoming and past exams in their respective tabs
            with tab3:
                st.subheader("🔄 Current Exams")
                if current_exams:
                    for exam in current_exams:
                        with st.expander(f"{exam['exam_name']} - {exam['course_code']}", expanded=True):
                            start_time = datetime.fromisoformat(exam['start_time'].replace('Z', '+00:00'))
                            end_time = datetime.fromisoformat(exam['end_time'].replace('Z', '+00:00'))
                            remaining = end_time - now
                            
                            st.write(f"⏱️ Time Remaining: {str(remaining).split('.')[0]}")
                            st.write(f"📍 Venue: {exam['venue']}")
                            st.write(f"👨‍🏫 Instructor: {exam['instructor']}")
                            
                            if st.button("📷 Monitor Now", key=f"monitor_{exam['id']}"):
                                st.session_state.current_page = "Live Monitoring"
                                st.session_state.current_exam_id = exam['id']
                                st.rerun()
                else:
                    st.info("No exams currently in progress")
                
                st.markdown("---")
                if st.button("➕ Configure New Exam", key="current_new_exam", use_container_width=True):
                    st.session_state.current_page = "Exam Configuration"
                    st.rerun()
            
            with tab4:
                st.subheader("📅 Upcoming Exams")
                if upcoming_exams:
                    for exam in upcoming_exams:
                        with st.expander(f"{exam['exam_name']} - {exam['course_code']}", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📆 Date: {exam['start_time'].split('T')[0]}")
                                st.write(f"🕒 Time: {exam['start_time'].split('T')[1][:5]}")
                                st.write(f"📍 Venue: {exam['venue']}")
                                st.write(f"👨‍🏫 Instructor: {exam['instructor']}")
                                st.write(f"👥 Total Students: {exam['total_students']}")
                                st.write(f"⏱️ Duration: {exam['duration']} minutes")
                            with col2:
                                action_col1, action_col2 = st.columns(2)
                                with action_col1:
                                    if st.button("🔄", key=f"reschedule_{exam['id']}", help="Reschedule Exam"):
                                        self._show_reschedule_form(exam)
                                with action_col2:
                                    if st.button("🗑️", key=f"del_{exam['id']}", help="Delete Exam", type="primary"):
                                        try:
                                            self.db_manager.supabase.table("exams").delete().eq('id', exam['id']).execute()
                                            st.success("Exam deleted successfully!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting exam: {str(e)}")
                else:
                    st.info("No upcoming exams scheduled")
                
                st.markdown("---")
                if st.button("➕ Configure New Exam", key="upcoming_new_exam", use_container_width=True):
                    st.session_state.current_page = "Exam Configuration"
                    st.rerun()
                    
            with tab5:
                st.subheader("📚 Past Exams")
                if past_exams:
                    for exam in past_exams:
                        with st.expander(f"{exam['exam_name']} - {exam['course_code']}", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📆 Date: {exam['start_time'].split('T')[0]}")
                                st.write(f"🕒 Time: {exam['start_time'].split('T')[1][:5]}")
                                st.write(f"📍 Venue: {exam['venue']}")
                                st.write(f"👨‍🏫 Instructor: {exam['instructor']}")
                                st.write(f"👥 Total Students: {exam['total_students']}")
                                st.write(f"⏱️ Duration: {exam['duration']} minutes")
                            with col2:
                                report_col, del_col = st.columns(2)
                                with report_col:
                                    if st.button("�", key=f"report_{exam['id']}", help="View Report"):
                                        st.session_state.current_page = "Reports"
                                        st.session_state.selected_exam_id = exam['id']
                                        st.rerun()
                                with del_col:
                                    if st.button("🗑️", key=f"del_past_{exam['id']}", help="Delete Exam", type="primary"):
                                        try:
                                            # Delete exam and related data from database
                                            self.db_manager.supabase.table("alerts").delete().eq('exam_id', exam['id']).execute()
                                            self.db_manager.supabase.table("detections").delete().eq('exam_id', exam['id']).execute()
                                            self.db_manager.supabase.table("exams").delete().eq('id', exam['id']).execute()
                                            st.success("Exam and related data deleted successfully!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting exam: {str(e)}")
                else:
                    st.info("No past exams found")
                
                st.markdown("---")
                if st.button("➕ Configure New Exam", key="past_new_exam", use_container_width=True):
                    st.session_state.current_page = "Exam Configuration"
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Error rendering scheduler: {str(e)}")
            logger.exception("Full stack trace:")  # This will log the full stack trace
            st.error(f"Failed to load exam scheduler: {str(e)}")

    def _check_schedule_conflicts(self, start_time: datetime, end_time: datetime, venue: str) -> bool:
        """Check for scheduling conflicts"""
        try:
            # Get all exams scheduled for the same venue
            venue_exams = self.db_manager.supabase.table("exams") \
                .select("id,start_time,end_time") \
                .eq("venue", venue) \
                .eq("status", "scheduled") \
                .execute()
                
            if not venue_exams.data:
                return False
                
            # Check for time overlaps
            for exam in venue_exams.data:
                exam_start = datetime.fromisoformat(exam['start_time'].replace('Z', '+00:00'))
                exam_end = datetime.fromisoformat(exam['end_time'].replace('Z', '+00:00'))
                
                if (start_time < exam_end and end_time > exam_start):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking schedule conflicts: {e}")
            return True  # Assume conflict on error to be safe
            
    def _show_reschedule_form(self, exam: Dict):
        """Show form for rescheduling an exam"""
        st.markdown("### 🔄 Reschedule Exam")
        
        with st.form(f"reschedule_exam_{exam['id']}"):
            date = st.date_input(
                "New Date",
                value=datetime.fromisoformat(exam['start_time']).date(),
                min_value=datetime.now().date()
            )
            time = st.time_input(
                "New Time",
                value=datetime.fromisoformat(exam['start_time']).time()
            )
            
            if st.form_submit_button("Save Changes", use_container_width=True):
                try:
                    new_start = datetime.combine(date, time)
                    new_end = new_start + timedelta(minutes=exam['duration'])
                    
                    if self._check_schedule_conflicts(new_start, new_end, exam['venue']):
                        st.error("Schedule conflict detected! Please choose another time.")
                        return
                        
                    # Prepare update data with all required fields
                    update_data = {
                        'exam_name': exam['exam_name'],
                        'course_code': exam['course_code'],
                        'department': exam['department'],
                        'instructor': exam['instructor'],
                        'total_students': exam['total_students'],
                        'venue': exam['venue'],
                        'duration': exam['duration'],
                        'exam_type': exam['exam_type'],  # This must exist from configuration
                        'status': exam['status'],
                        'start_time': new_start.isoformat(),
                        'end_time': new_end.isoformat(),
                        'last_updated': datetime.now(pytz.UTC).isoformat()
                    }
                    
                    self.db_manager.update_exam(
                        exam['id'],
                        update_data
                    )
                    
                    st.success("Exam rescheduled successfully!")
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Error rescheduling exam: {e}")
                    st.error(f"Failed to reschedule exam: {str(e)}")
                    
    def _cancel_exam(self, exam_id: str):
        """Cancel a scheduled exam"""
        try:
            # Get current exam data first
            current_exam = self.db_manager.get_exam(exam_id)
            if not current_exam:
                raise ValueError("Exam not found")
                
            # Prepare update data with all required fields
            update_data = {
                'exam_name': current_exam['exam_name'],
                'course_code': current_exam['course_code'],
                'department': current_exam['department'],
                'instructor': current_exam['instructor'],
                'total_students': current_exam['total_students'],
                'venue': current_exam['venue'],
                'duration': current_exam['duration'],
                'exam_type': current_exam['exam_type'],  # This must exist from configuration
                'status': 'configured',
                'start_time': None,
                'end_time': None,
                'last_updated': datetime.now(pytz.UTC).isoformat()
            }
            
            self.db_manager.update_exam(
                exam_id,
                update_data
            )
            st.success("Exam cancelled successfully!")
        except Exception as e:
            logger.error(f"Error cancelling exam: {e}")
            st.error(f"Failed to cancel exam: {str(e)}")

