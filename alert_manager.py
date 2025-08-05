import cv2
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

class AlertManager:
    """Handles exam monitoring alerts and statistics"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.alert_counters = {}  # Track alert counts per exam
        self.last_alert_time = {}  # Track last alert time per exam and type
        self.alert_cooldown = timedelta(seconds=30)  # Minimum time between alerts
        self.ensure_alert_directory()
    
    def ensure_alert_directory(self):
        """Ensure the alerts directory exists"""
        os.makedirs("cheating_detections", exist_ok=True)
    
    def create_alert(self, exam_id: str, alert_type: str, confidence: float, frame=None):
        """Create a new alert and save it to the database"""
        try:
            # Check cooldown period
            now = datetime.now()
            alert_key = (exam_id, alert_type)
            if alert_key in self.last_alert_time:
                if now - self.last_alert_time[alert_key] < self.alert_cooldown:
                    return  # Skip alert if within cooldown period
            
            # Initialize counter for this exam if needed
            if exam_id not in self.alert_counters:
                self.alert_counters[exam_id] = {
                    "face_detection": 0,
                    "object_detection": 0,
                    "motion_detection": 0
                }
            
            # Update counters and timestamps
            self.alert_counters[exam_id][alert_type] += 1
            self.last_alert_time[alert_key] = now
            
            # Save the frame if provided
            image_path = None
            if frame is not None:
                timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                filename = f"{alert_type}_{timestamp_str}_{confidence:.2f}.jpg"
                image_path = os.path.join("cheating_detections", filename)
                cv2.imwrite(image_path, frame)
            
            # Create alert record
            alert_data = {
                "exam_id": exam_id,
                "alert_type": alert_type,
                "confidence": confidence,
                "timestamp": now.isoformat(),
                "image_path": image_path
            }
            
            # Save to database
            try:
                result = self.db_manager.supabase.table("exam_alerts").insert(alert_data).execute()
                if not result.data:
                    logger.error("Failed to save alert to database")
            except Exception as e:
                logger.error(f"Database error saving alert: {e}")
                
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    def get_exam_statistics(self, exam_id: str) -> Dict[str, int]:
        """Get current alert statistics for an exam"""
        if exam_id not in self.alert_counters:
            return {
                "face_detection": 0,
                "object_detection": 0,
                "motion_detection": 0
            }
        return self.alert_counters[exam_id].copy()  # Return a copy to prevent modification
    
    def get_exam_alerts(self, exam_id: str, limit: int = 10) -> List[Dict]:
        """Get recent alerts for an exam"""
        try:
            result = self.db_manager.supabase.table("exam_alerts").select(
                "*"
            ).eq("exam_id", exam_id).order("timestamp", desc=True).limit(limit).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error getting exam alerts: {e}")
            return []