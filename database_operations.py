import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class DatabaseOperations:
    def save_exam_report(self, report_summary: Dict) -> Dict:
        """Save a generated exam report summary to the exam_reports table (for compatibility with report generator)."""
        try:
            # Use create_exam_report for full compatibility
            return self.create_exam_report(
                exam_id=report_summary.get('exam_id'),
                report_type=report_summary.get('status', 'generated'),
                report_format='summary',
                report_data=report_summary,
                file_path=None,
                generated_by=None,
                version=report_summary.get('version', '1.0')
            )
        except Exception as e:
            logger.error(f"Failed to save exam report: {e}")
            raise

    def export_report_data(self, exam_id: str) -> Dict:
        """Export a comprehensive report for an exam, including metrics, alerts, and detections."""
        try:
            metrics = self.get_exam_metrics(exam_id)
            alerts = self.get_alerts_by_exam(exam_id, include_reviewed=True)
            detections = self.get_detections_by_exam(exam_id)
            reports = self.get_exam_reports(exam_id)
            return {
                'metrics': metrics,
                'alerts': alerts,
                'detections': detections,
                'reports': reports
            }
        except Exception as e:
            logger.error(f"Failed to export report data: {e}")
            return {}
    def __init__(self, supabase_client):
        """Initialize DatabaseOperations with Supabase client."""
        self.supabase = supabase_client

    # Detection Management Methods
    def create_detection(self, exam_id: str, detection_type: str, confidence: float, 
                        people_count: Optional[int] = None, details: Optional[Dict] = None) -> Dict:
        """
        Create a new detection record.
        
        Args:
            exam_id: UUID of the exam
            detection_type: Type of detection (e.g., 'cheating', 'normal')
            confidence: Confidence score of the detection
            people_count: Number of people detected (optional)
            details: Additional detection details (optional)
            
        Returns:
            Dict containing the created detection record
        """
        try:
            detection_data = {
                'exam_id': exam_id,
                'detection_type': detection_type,
                'confidence': confidence,
                'people_count': people_count,
                'details': details
            }
            result = self.supabase.table('detections').insert(detection_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create detection: {e}")
            raise

    def get_detections_by_exam(self, exam_id: str) -> List[Dict]:
        """Get all detections for a specific exam."""
        try:
            result = self.supabase.table('detections')\
                .select('*')\
                .eq('exam_id', exam_id)\
                .order('timestamp', desc=True)\
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get detections: {e}")
            raise

    # Alert Management Methods
    def create_alert(self, exam_id: str, alert_type: str, confidence: float, 
                    evidence_path: Optional[str] = None, details: Optional[Dict] = None) -> Dict:
        """Create a new alert record."""
        try:
            alert_data = {
                'exam_id': exam_id,
                'alert_type': alert_type,
                'confidence': confidence,
                'evidence_path': evidence_path,
                'details': details,
                'reviewed': False
            }
            result = self.supabase.table('alerts').insert(alert_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise

    def get_alerts_by_exam(self, exam_id: str, include_reviewed: bool = False) -> List[Dict]:
        """Get alerts for a specific exam."""
        try:
            query = self.supabase.table('alerts')\
                .select('*')\
                .eq('exam_id', exam_id)
            
            if not include_reviewed:
                query = query.eq('reviewed', False)
                
            result = query.order('timestamp', desc=True).execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            raise

    def update_alert(self, alert_id: str, update_data: Dict) -> Dict:
        """Update an alert record."""
        try:
            result = self.supabase.table('alerts')\
                .update(update_data)\
                .eq('id', alert_id)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to update alert: {e}")
            raise

    # Exam Metrics Management Methods
    def create_or_update_metrics(self, exam_id: str, metrics_data: Dict) -> Dict:
        """Create or update exam metrics."""
        try:
            metrics = {
                'exam_id': exam_id,
                'total_detections': metrics_data.get('total_detections', 0),
                'total_cheating': metrics_data.get('total_cheating', 0),
                'total_good_behavior': metrics_data.get('total_good_behavior', 0),
                'cheating_percentage': (metrics_data.get('total_cheating', 0) / 
                    metrics_data.get('total_detections', 1) * 100 if metrics_data.get('total_detections', 0) > 0 else 0),
                'good_behavior_percentage': (metrics_data.get('total_good_behavior', 0) / 
                    metrics_data.get('total_detections', 1) * 100 if metrics_data.get('total_detections', 0) > 0 else 0),
                'mean_confidence': metrics_data.get('mean_confidence', 0),
                'median_confidence': metrics_data.get('median_confidence', 0),
                'max_confidence': metrics_data.get('max_confidence', 0),
                'min_confidence': metrics_data.get('min_confidence', 0),
                'std_deviation': metrics_data.get('std_deviation', 0),
                'peak_cheating_hour': metrics_data.get('peak_cheating_hour'),
                'lowest_cheating_hour': metrics_data.get('lowest_cheating_hour'),
                'alert_frequency': metrics_data.get('alert_frequency', 0),
                'average_response_time': metrics_data.get('average_response_time', 0),
                'total_alerts': metrics_data.get('total_alerts', 0),
                'resolved_alerts': metrics_data.get('resolved_alerts', 0),
                'alert_resolution_rate': (metrics_data.get('resolved_alerts', 0) / 
                    metrics_data.get('total_alerts', 1) * 100 if metrics_data.get('total_alerts', 0) > 0 else 0),
                'attendance_rate': metrics_data.get('attendance_rate', 0),
                'average_students_present': metrics_data.get('average_students_present', 0),
                'monitoring_duration': metrics_data.get('monitoring_duration'),
                'metrics_data': metrics_data.get('additional_data', {})
            }
            
            result = self.supabase.table('exam_metrics')\
                .upsert(metrics)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create/update exam metrics: {e}")
            raise

    def get_exam_metrics(self, exam_id: str) -> Dict:
        """Get metrics for a specific exam."""
        try:
            result = self.supabase.table('exam_metrics')\
                .select('*')\
                .eq('exam_id', exam_id)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get exam metrics: {e}")
            raise

    # Report Management Methods
    def create_exam_report(self, exam_id: str, report_type: str, report_format: str, 
                          report_data: Dict, file_path: Optional[str] = None, 
                          generated_by: Optional[str] = None, version: str = "1.0") -> Dict:
        """Create a new exam report."""
        try:
            report = {
                'exam_id': exam_id,
                'report_type': report_type,
                'report_format': report_format,
                'report_data': report_data,
                'file_path': file_path,
                'generated_by': generated_by,
                'report_version': version,
                'is_final': False,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'version': version
                }
            }
            
            result = self.supabase.table('exam_reports')\
                .insert(report)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create exam report: {e}")
            raise

    def get_exam_reports(self, exam_id: str, report_type: Optional[str] = None) -> List[Dict]:
        """Get reports for a specific exam."""
        try:
            query = self.supabase.table('exam_reports')\
                .select('*')\
                .eq('exam_id', exam_id)
                
            if report_type:
                query = query.eq('report_type', report_type)
                
            result = query.order('generated_at', desc=True).execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get exam reports: {e}")
            raise

    def finalize_report(self, report_id: str) -> Dict:
        """Mark a report as final."""
        try:
            result = self.supabase.table('exam_reports')\
                .update({'is_final': True})\
                .eq('id', report_id)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to finalize report: {e}")
            raise

    # Utility Methods
    def get_dashboard_metrics(self) -> Dict:
        """Get aggregated metrics for the dashboard."""
        try:
            # Get total exams count
            exams_result = self.supabase.table('exams')\
                .select('status')\
                .execute()
            
            exams = exams_result.data if exams_result.data else []
            
            # Get recent alerts
            alerts_result = self.supabase.table('alerts')\
                .select('*')\
                .eq('reviewed', False)\
                .order('timestamp', desc=True)\
                .limit(5)\
                .execute()
            
            # Get upcoming exams
            upcoming_exams_result = self.supabase.table('exams')\
                .select('*')\
                .eq('status', 'scheduled')\
                .order('start_time')\
                .limit(5)\
                .execute()
            
            metrics = {
                'total_exams': len(exams),
                'active_exams': len([e for e in exams if e['status'] == 'running']),
                'completed_exams': len([e for e in exams if e['status'] == 'completed']),
                'total_alerts': len(alerts_result.data) if alerts_result.data else 0,
                'pending_reviews': len([a for a in (alerts_result.data or []) if not a.get('reviewed')]),
                'recent_alerts': alerts_result.data if alerts_result.data else [],
                'upcoming_exams': upcoming_exams_result.data if upcoming_exams_result.data else []
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            return {
                'total_exams': 0,
                'active_exams': 0,
                'completed_exams': 0,
                'total_alerts': 0,
                'pending_reviews': 0,
                'recent_alerts': [],
                'upcoming_exams': []
            }
