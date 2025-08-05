import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Handles exam report generation and visualization"""
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def render_report_page(self):
        """Render the report generation page"""
        st.title("📊 Exam Reports")
        
        try:
            # Get completed exams
            completed_exams = self.db_manager.get_exams_by_status("completed")
            
            if not completed_exams:
                st.info("No completed exams found. Reports will be available after exams are completed.")
                return
                
            # Exam selection
            selected_exam = st.selectbox(
                "Select Exam",
                options=completed_exams,
                format_func=lambda x: f"{x['exam_name']} ({x['course_code']}) - {x['end_time'].split('T')[0]}"
            )
            
            if selected_exam:
                self._render_exam_report(selected_exam)
            
        except Exception as e:
            logger.error(f"Error rendering report page: {e}")
            st.error("Failed to load exam reports. Please try again later.")
            
    def _render_exam_report(self, exam_data: dict):
        """Generate and display report for a specific exam"""
        try:
            # Get exam data
            exam_id = exam_data['id']
            detections = self.db_manager.get_exam_detections(exam_id)
            alerts = self.db_manager.get_exam_alerts(exam_id)
            
            # Report Header
            st.header(f"📑 Report: {exam_data['exam_name']}")
            
            # Exam Overview
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
            
            # Key Metrics
            st.subheader("📊 Key Metrics")
            
            # Calculate metrics
            total_detections = len(detections)
            total_alerts = len(alerts)
            unique_incidents = len({alert['timestamp'].split('T')[0] for alert in alerts})
            alert_rate = (total_alerts / total_detections * 100) if total_detections > 0 else 0
            
            # Display metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Detections", total_detections)
            with metric_cols[1]:
                st.metric("Total Alerts", total_alerts)
            with metric_cols[2]:
                st.metric("Unique Incidents", unique_incidents)
            with metric_cols[3]:
                st.metric("Alert Rate", f"{alert_rate:.1f}%")
            
            # Timeline Analysis
            st.subheader("📈 Timeline Analysis")
            if alerts:
                # Convert timestamps to datetime
                df_alerts = pd.DataFrame(alerts)
                df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
                
                # Group by hour
                hourly_alerts = df_alerts.groupby(df_alerts['timestamp'].dt.hour).size()
                
                # Create line plot
                fig, ax = plt.subplots(figsize=(10, 6))
                hourly_alerts.plot(kind='line', marker='o')
                plt.title('Alert Distribution Over Time')
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Alerts')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Incident Types
            if alerts:
                st.subheader("🔍 Incident Analysis")
                incident_types = pd.DataFrame(alerts)['alert_type'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    # Pie chart of incident types
                    fig, ax = plt.subplots()
                    plt.pie(incident_types.values, labels=incident_types.index, autopct='%1.1f%%')
                    plt.title('Incident Type Distribution')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Bar chart of incident severity
                    if 'confidence' in df_alerts.columns:
                        severity_bins = pd.cut(df_alerts['confidence'], 
                                            bins=[0, 0.3, 0.6, 0.8, 1.0],
                                            labels=['Low', 'Medium', 'High', 'Critical'])
                        severity_dist = severity_bins.value_counts()
                        
                        fig, ax = plt.subplots()
                        severity_dist.plot(kind='bar')
                        plt.title('Incident Severity Distribution')
                        plt.xlabel('Severity Level')
                        plt.ylabel('Count')
                        st.pyplot(fig)
                        plt.close()
            
            # Evidence Gallery
            st.subheader("🖼️ Evidence Gallery")
            alert_images = [alert for alert in alerts if alert.get('evidence_path')]
            
            if alert_images:
                # Display images in a grid
                cols = st.columns(3)
                for idx, alert in enumerate(alert_images):
                    if os.path.exists(alert['evidence_path']):
                        with cols[idx % 3]:
                            st.image(
                                alert['evidence_path'],
                                caption=f"Alert at {alert['timestamp'].split('T')[1][:8]} ({alert['confidence']:.2f})",
                                use_column_width=True
                            )
            else:
                st.info("No evidence images available for this exam.")
            
            # Export Options
            st.subheader("📤 Export Report")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download CSV", use_container_width=True):
                    df = pd.DataFrame(alerts)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Report CSV",
                        csv,
                        f"exam_report_{exam_data['exam_name']}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        key='download-csv',
                        use_container_width=True
                    )
                    
            with col2:
                if st.button("Download PDF", use_container_width=True):
                    st.info("PDF export functionality coming soon!")
                    
        except Exception as e:
            logger.error(f"Error generating exam report: {e}")
            st.error("Failed to generate exam report. Please try again later.")
