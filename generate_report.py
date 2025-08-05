import streamlit as st
import datetime
import streamlit as st
import datetime
import pandas as pd
from fpdf import FPDF
import base64
from io import BytesIO
import time
import os


def configure_examination():
    """Configure the examination settings"""
    st.title("Examination Configuration")
    st.sidebar.markdown("### Please fill all the required fields to start monitoring the exam.")
    
    # Example configuration options
    exam_type = st.text_input("Exam Type", placeholder="eg., Midsemester, End of Semester")
    exam_department = st.text_input("Department", placeholder="e.g., Computer Science")
    exam_name = st.text_input("Exam Name", placeholder="e.g., Data Structures and Algorithms")
    exam_instructor = st.text_input("Instructor Name", placeholder="e.g., Dr. John Doe")
    venue = st.text_input("Venue", placeholder="e.g., Room 101")
    exam_time = st.time_input("Exam Time", value=datetime.time(9, 0))  # Default to 9:00 AM
    exam_duration = st.number_input("Exam Duration (minutes)", min_value=30, max_value=180, value=60)  # Default to 60 minutes
    exam_date = st.date_input("Exam Date", value=datetime.date.today())
    exam_duration = st.number_input("Exam Duration (minutes)", min_value=30, max_value=180, value=60)
    
    if st.button("Save Configuration"):
        # Save the configuration (this is just a placeholder)
        st.success(f"Configuration saved for {exam_name} on {exam_date} with duration {exam_duration} minutes.")


def schedule_exam():
    """Schedule the exam"""
    st.title("Exam Scheduling")
    st.sidebar.markdown("### Please fill all the required fields to schedule the exam.")
    
    # Example scheduling options
    exam_date = st.date_input("Exam Date", value=datetime.date.today())
    exam_time = st.time_input("Exam Time", value=datetime.time(9, 0))  # Default to 9:00 AM
    exam_duration = st.number_input("Exam Duration (minutes)", min_value=30, max_value=180, value=60)  # Default to 60 minutes
    
    if st.button("Schedule Exam"):
        # Schedule the exam (this is just a placeholder)
        st.success(f"Exam scheduled on {exam_date} at {exam_time} for {exam_duration} minutes.")




# PDF Generation Function
def create_pdf(exam_data, metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Examination Report on {exam_data['exam_name']}", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {exam_data['exam_date'].strftime('%A, %B %d, %Y')}", ln=1)
    pdf.ln(10)
    
    # Introduction
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Introduction", ln=1)
    pdf.set_font("Arial", size=12)
    intro_text = (
        f"At {exam_data['exam_time']}, {exam_data['invigilator_name']} supervised {metrics['total_students']} "
        f"students from the {exam_data['exam_department']} department in {exam_data['venue']} for a "
        f"duration of {exam_data['duration']} minutes. This report summarizes the behavioral analysis and "
        "monitoring results from the examination session."
    )
    pdf.multi_cell(0, 10, txt=intro_text)
    pdf.ln(5)
    
    # Key Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Key Examination Metrics", ln=1)
    pdf.set_font("Arial", size=12)
    
    # Metrics Table
    col_widths = [90, 90]
    pdf.set_fill_color(200, 220, 255)
    
    # Header
    pdf.cell(col_widths[0], 10, "Metric", border=1, fill=True)
    pdf.cell(col_widths[1], 10, "Value", border=1, fill=True, ln=1)
    
    # Rows
    metrics_data = [
        ("Total Students", str(metrics['total_students'])),
        ("Total Detections", str(metrics['total_detection_count'])),
        ("Cheating Incidents", str(metrics['total_cheating_incidents'])),
        ("Good Behavior", str(metrics['total_good_behavior'])),
        ("Cheating Rate", f"{metrics['cheating_percentage']:.1f}%"),
        ("Good Behavior Rate", f"{metrics['good_behavior_percentage']:.1f}%"),
        ("Detections per Student", f"{metrics['detections_per_student']:.1f}"),
        ("Cheating per Student", f"{metrics['cheating_per_student']:.2f}"),
    ]
    
    for row in metrics_data:
        pdf.cell(col_widths[0], 10, row[0], border=1)
        pdf.cell(col_widths[1], 10, row[1], border=1, ln=1)
    
    pdf.ln(10)
    
    # Detailed Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Detailed Analysis", ln=1)
    pdf.set_font("Arial", size=12)
    
    analysis_text = (
        f"The examination monitoring system recorded {metrics['total_detection_count']} suspicious events "
        f"across {metrics['total_students']} students, averaging {metrics['detections_per_student']:.1f} "
        f"detections per student. After careful review, {metrics['total_cheating_incidents']} confirmed "
        f"cheating incidents were identified, affecting {metrics['cheating_percentage']:.1f}% of the examinees.\n\n"
        f"On a positive note, {metrics['total_good_behavior']} students ({metrics['good_behavior_percentage']:.1f}%) "
        f"demonstrated exemplary behavior throughout the examination with zero suspicious activities detected. "
        f"The average cheating incidents per student stood at {metrics['cheating_per_student']:.2f}."
    )
    pdf.multi_cell(0, 10, txt=analysis_text)
    pdf.ln(5)
    
    # Conclusion and Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Conclusion", ln=1)
    pdf.set_font("Arial", size=12)
    
    if metrics['cheating_percentage'] < 5:
        conclusion = (
            "The examination was conducted with excellent discipline. The low cheating incidence rate indicates "
            "effective proctoring and student adherence to examination rules. The high percentage of good "
            "behavior suggests a positive examination environment."
        )
    elif 5 <= metrics['cheating_percentage'] < 15:
        conclusion = (
            "The examination experienced moderate behavioral challenges. While most students complied with "
            "regulations, the cheating incidents warrant attention to prevent future occurrences."
        )
    else:
        conclusion = (
            "The examination faced significant behavioral issues. The high cheating incidence rate suggests "
            "the need for immediate intervention to maintain academic standards."
        )
    
    pdf.multi_cell(0, 10, txt=conclusion)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Recommendations", ln=1)
    pdf.set_font("Arial", size=12)
    
    if metrics['cheating_percentage'] < 5:
        recommendation = (
            "Maintain current proctoring standards. Consider recognizing students with perfect behavior "
            "to encourage continued compliance."
        )
    elif 5 <= metrics['cheating_percentage'] < 15:
        recommendation = (
            "Implement additional pre-exam briefings on academic integrity. Consider increasing invigilator "
            "presence in areas where most incidents were detected."
        )
    else:
        recommendation = (
            "Conduct a thorough review of examination procedures. Implement stricter proctoring measures "
            "and consider academic integrity workshops before future exams."
        )
    
    pdf.multi_cell(0, 10, txt=recommendation)
    
    return pdf

def generate_download_link(pdf):
    """Generate a download link for the PDF"""
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="examination_report.pdf">Download PDF Report</a>'
    return href

def generate_report():
    """Generate a comprehensive examination report with PDF download"""
    st.title("Examination Report")
    
    # Sample data - replace with your actual data
    exam_data = {
        'exam_name': "Data Structures and Algorithms",
        'exam_time': "09:00 AM",
        'invigilator_name': "Dr. John Doe",
        'exam_department': "Computer Science",
        'venue': "Room 101",
        'duration': 120,  # 2 hours in minutes
        'exam_date': datetime.date.today()
    }
    
    metrics = {
        'total_students': 45,
        'total_detection_count': 112,
        'total_cheating_incidents': 18,
        'total_good_behavior': 39,
    }
    
    # Calculate derived metrics
    metrics['cheating_percentage'] = (metrics['total_cheating_incidents'] / metrics['total_students']) * 100
    metrics['good_behavior_percentage'] = (metrics['total_good_behavior'] / metrics['total_students']) * 100
    metrics['detections_per_student'] = metrics['total_detection_count'] / metrics['total_students']
    metrics['cheating_per_student'] = metrics['total_cheating_incidents'] / metrics['total_students']
    
    # Display the report in Streamlit
    st.markdown(f"## Examination Report on {exam_data['exam_name']}")
    st.markdown(f"**Date:** {exam_data['exam_date'].strftime('%A, %B %d, %Y')}")
    st.markdown("---")
    
    # Introduction
    st.markdown("### Introduction")
    st.markdown(
        f"At {exam_data['exam_time']}, {exam_data['invigilator_name']} supervised {metrics['total_students']} "
        f"students from the {exam_data['exam_department']} department in {exam_data['venue']} for a "
        f"duration of {exam_data['duration']}. This report summarizes the behavioral analysis and "
        "monitoring results from the examination session."
    )
    
    # Generate the PDF
    pdf = create_pdf(exam_data, metrics)
    
    # Create download link
    st.markdown("---")
    st.markdown("### PDF Report Generation")
    st.markdown(generate_download_link(pdf), unsafe_allow_html=True)

# Main app
if __name__ == "__main__":
    generate_report()    







    

               