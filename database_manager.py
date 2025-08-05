import logging
import os
from typing import List, Dict, Optional
from datetime import datetime
from supabase import Client, create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_supabase() -> Client:
    """Initialize Supabase client"""
    # Get Supabase credentials
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in .env file")
        
    return create_client(url, key)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        
    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for the dashboard"""
        try:
            # Get exam counts by status
            exams = self.supabase.table("exams").select("*").execute()
            exam_data = exams.data if exams else []
            
            # Get alerts
            alerts = self.supabase.table("alerts").select("*").execute()
            alert_data = alerts.data if alerts else []
            
            # Calculate metrics
            metrics = {
                'total_exams': len(exam_data),
                'active_exams': len([e for e in exam_data if e['status'] == 'running']),
                'completed_exams': len([e for e in exam_data if e['status'] == 'completed']),
                'total_alerts': len(alert_data),
                'pending_reviews': len([a for a in alert_data if not a.get('reviewed', False)]),
                'recent_alerts': sorted(
                    [a for a in alert_data if not a.get('reviewed', False)],
                    key=lambda x: x['timestamp'],
                    reverse=True
                )[:5],
                'upcoming_exams': sorted(
                    [e for e in exam_data if e['status'] == 'scheduled'],
                    key=lambda x: x['start_time']
                )[:5]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            raise
            
    def get_exam(self, exam_id: str) -> Optional[Dict]:
        """Get exam by ID"""
        try:
            result = self.supabase.table("exams").select("*").eq("id", exam_id).execute()
            return result.data[0] if result and result.data else None
        except Exception as e:
            logger.error(f"Error getting exam: {e}")
            raise
    
    def get_exams_by_status(self, status: str) -> List[Dict]:
        """Get all exams with specific status"""
        try:
            result = self.supabase.table("exams") \
                .select("*") \
                .eq("status", status) \
                .order("start_time", desc=False) \
                .execute()
            return result.data if result else []
        except Exception as e:
            logger.error(f"Error getting exams by status: {e}")
            raise
            
    def get_exam_detections(self, exam_id: str) -> List[Dict]:
        """Get all detections for a specific exam"""
        try:
            result = self.supabase.table("detections") \
                .select("*") \
                .eq("exam_id", exam_id) \
                .order("timestamp", desc=False) \
                .execute()
            return result.data if result else []
        except Exception as e:
            logger.error(f"Error getting exam detections: {e}")
            raise
            
    def get_exam_alerts(self, exam_id: str) -> List[Dict]:
        """Get all alerts for a specific exam"""
        try:
            result = self.supabase.table("alerts") \
                .select("*") \
                .eq("exam_id", exam_id) \
                .order("timestamp", desc=False) \
                .execute()
            return result.data if result else []
        except Exception as e:
            logger.error(f"Error getting exam alerts: {e}")
            raise
            
    def update_exam(self, exam_id: str, updates: Dict) -> Dict:
        """Update exam data"""
        try:
            result = self.supabase.table("exams") \
                .update(updates) \
                .eq("id", exam_id) \
                .execute()
            return result.data[0] if result and result.data else {}
        except Exception as e:
            logger.error(f"Error updating exam: {e}")
            raise
            
    def update_alert(self, alert_id: str, updates: Dict) -> Dict:
        """Update alert data"""
        try:
            result = self.supabase.table("alerts") \
                .update(updates) \
                .eq("id", alert_id) \
                .execute()
            return result.data[0] if result and result.data else {}
        except Exception as e:
            logger.error(f"Error updating alert: {e}")
            raise
            
    def add_detection(self, detection_data: Dict) -> Dict:
        """Add a new detection record"""
        try:
            result = self.supabase.table("detections").insert(detection_data).execute()
            return result.data[0] if result and result.data else {}
        except Exception as e:
            logger.error(f"Error adding detection: {e}")
            raise
            
    def add_alert(self, alert_data: Dict) -> Dict:
        """Add a new alert record"""
        try:
            result = self.supabase.table("alerts").insert(alert_data).execute()
            return result.data[0] if result and result.data else {}
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            raise
