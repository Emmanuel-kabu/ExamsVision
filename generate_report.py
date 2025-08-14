import streamlit as st
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import base64
import matplotlib.pyplot as plt


class PDFReport(FPDF):
    """Custom PDF report generator with formatted sections"""
    
    def header(self):
        """Add report header"""
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, "Examination Monitoring Report", ln=1, align="C")
        self.ln(5)
    
    def chapter_title(self, title):
        """Add styled chapter title"""
        self.set_fill_color(50, 50, 50)
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", 'B', 14)
        self.cell(0, 8, title, ln=1, align="L", fill=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)
    
    def chapter_body(self, text):
        """Add chapter content"""
        self.set_font("Arial", size=12)
        self.multi_cell(0, 8, text)
        self.ln()
    
    def add_image(self, img_bytes, width=100, height=70):
        """Add image to PDF from bytes"""
        temp = BytesIO(img_bytes)
        self.image(temp, x=None, y=None, w=width, h=height)
        self.ln(5)


def generate_pie_chart(metrics):
    """Generate pie chart visualization"""
    labels = ['Good Behavior', 'Cheating Incidents']
    sizes = [metrics['total_good_behavior'], metrics['total_cheating_incidents']]
    colors = ['#4CAF50', '#F44336']
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, 
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def generate_bar_chart(metrics):
    """Generate bar chart visualization"""
    categories = ['Detections per Student', 'Cheating per Student']
    values = [metrics['detections_per_student'], metrics['cheating_per_student']]
    colors = ['#2196F3', '#FF9800']
    
    fig, ax = plt.subplots()
    ax.bar(categories, values, color=colors)
    ax.set_ylabel("Average Count")
    ax.set_title("Incident Metrics")
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def create_pdf(exam_data, metrics):
    """Generate complete PDF report"""
    pdf = PDFReport()
    pdf.add_page()
    
    # Report title section
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, f"{exam_data['exam_name']}", ln=1, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, 
             f"Date: {exam_data['exam_date'].strftime('%A, %B %d, %Y')} | "
             f"Time: {exam_data['exam_time']}", 
             ln=1, align="C")
    pdf.ln(10)
    
    # Introduction section
    pdf.chapter_title("Introduction")
    intro_text = (
        f"On {exam_data['exam_date'].strftime('%B %d, %Y')} at {exam_data['exam_time']}, "
        f"{exam_data['invigilator_name']} supervised {metrics['total_students']} candidates "
        f"from the {exam_data['exam_department']} department in {exam_data['venue']}. "
        f"This {exam_data['duration']}-minute assessment was closely monitored using "
        "AI-powered proctoring tools to ensure academic integrity."
    )
    pdf.chapter_body(intro_text)
    
    # Metrics table
    pdf.chapter_title("Key Performance Metrics")
    col_widths = [90, 90]
    
    # Table header
    pdf.set_fill_color(220, 230, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(col_widths[0], 8, "Metric", border=1, fill=True)
    pdf.cell(col_widths[1], 8, "Value", border=1, ln=1, fill=True)
    
    # Table content
    pdf.set_font("Arial", size=12)
    table_data = [
        ("Total Students", str(metrics['total_students'])),
        ("Total Detections", str(metrics['total_detection_count'])),
        ("Cheating Incidents", str(metrics['total_cheating_incidents'])),
        ("Good Behavior", str(metrics['total_good_behavior'])),
        ("Cheating Rate", f"{metrics['cheating_percentage']:.1f}%"),
        ("Good Behavior Rate", f"{metrics['good_behavior_percentage']:.1f}%"),
        ("Detections per Student", f"{metrics['detections_per_student']:.1f}"),
        ("Cheating per Student", f"{metrics['cheating_per_student']:.2f}"),
    ]
    
    fill = False
    for row in table_data:
        pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_widths[0], 8, row[0], border=1, fill=True)
        pdf.cell(col_widths[1], 8, row[1], border=1, fill=True, ln=1)
        fill = not fill
    
    pdf.ln(5)
    
    # Charts section
    pdf.chapter_title("Visual Analysis")
    pdf.chapter_body("Below are visual summaries of the examination behavior and incident distribution.")
    pdf.add_image(generate_pie_chart(metrics), w=90, h=70)
    pdf.add_image(generate_bar_chart(metrics), w=90, h=70)
    
    # Conclusion section
    pdf.chapter_title("Conclusion & Recommendations")
    
    if metrics['cheating_percentage'] < 5:
        conclusion = "This session demonstrated outstanding discipline with minimal misconduct."
        recommendation = "Maintain current protocols and reward students with perfect conduct."
    elif metrics['cheating_percentage'] < 15:
        conclusion = "Moderate levels of misconduct were observed."
        recommendation = "Enhance vigilance and issue pre-exam integrity reminders."
    else:
        conclusion = "High levels of misconduct threaten academic integrity."
        recommendation = "Strengthen proctoring protocols and introduce integrity workshops."
    
    pdf.chapter_body(f"Conclusion: {conclusion}\n\nRecommendations: {recommendation}")
    
    return pdf


def generate_download_link(pdf):
    """Create download link for the PDF"""
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="examination_report.pdf">📄 Download PDF Report</a>'


def generate_report():
    """Main function to generate and display the report"""
    st.title("📄 Examination Report Generator")
    
    # Default exam data
    default_exam_data = {
        'exam_name': "Data Structures and Algorithms",
        'exam_time': "09:00 AM",
        'invigilator_name': "Dr. John Doe",
        'exam_department': "Computer Science",
        'venue': "Room 101",
        'duration': 120,
        'exam_date': datetime.today()
    }
    
    exam_data = st.session_state.get("exam_info", default_exam_data)
    metrics = st.session_state.get("alert_metrics")
    
    if not metrics:
        st.error("⚠ No monitoring metrics found. Please run the monitoring system first.")
        return
    
    # Generate and display PDF
    pdf = create_pdf(exam_data, metrics)
    st.markdown(generate_download_link(pdf), unsafe_allow_html=True)


if __name__ == "__main__":
    generate_report()





    

               