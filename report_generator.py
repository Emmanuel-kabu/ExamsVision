import streamlit as st
from generate_report import PDFReport, create_pdf, generate_download_link
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def render_report_page():
    st.title("📊 Exam Reports")

    try:
        completed_exams = st.session_state.get("completed_exams", [])
        if not completed_exams:
            st.info("No completed exams found.")
            return

        selected_exam = st.selectbox(
            "Select Exam",
            options=completed_exams,
            format_func=lambda x: f"{x['exam_name']} ({x['course_code']}) - {x['end_time'].split('T')[0]}"
        )

        if not selected_exam:
            return

        exam_id = selected_exam["id"]

        exam_data = st.session_state.get(f"exam_info_{exam_id}")
        metrics = st.session_state.get(f"exam_metrics_{exam_id}")

        if not exam_data or not metrics:
            st.warning("Report data not available yet. Please generate the report first.")
            return

        # 📋 Overview
        with st.expander("📋 Exam Overview", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    - **Course:** {exam_data['course_code']}
                    - **Department:** {exam_data['department']}
                    - **Instructor:** {exam_data['instructor']}
                    - **Venue:** {exam_data['venue']}
                """)
            with col2:
                st.markdown(f"""
                    - **Date:** {exam_data['start_time'].split('T')[0]}
                    - **Duration:** {exam_data['duration']} minutes
                    - **Total Students:** {exam_data['total_students']}
                    - **Status:** {exam_data['status'].title()}
                """)

        # 📊 Metrics
        st.subheader("📊 Key Metrics")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Detections", metrics['total_detection_count'])
        with metric_cols[1]:
            st.metric("Total Alerts", metrics['total_cheating_incidents'])
        with metric_cols[2]:
            st.metric("Unique Incidents", metrics['total_good_behavior'])
        with metric_cols[3]:
            st.metric("Alert Rate", f"{metrics['cheating_percentage']:.1f}%")

        # 📄 PDF download
        st.subheader("📄 Download Report")
        if st.button("Generate PDF Report"):
            pdf = create_pdf(exam_data, metrics)  # This returns a PDFReport object
            st.markdown(generate_download_link(pdf), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to load exam report: {e}")
