import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from supabase import Client, create_client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SupabaseManager:
    """Handles Supabase client initialization and connection management"""
    
    @staticmethod
    def initialize_client() -> Client:
        """
        Initialize and return a Supabase client.
        
        Returns:
            Client: Initialized Supabase client
            
        Raises:
            ValueError: If environment variables are missing or invalid
            ConnectionError: If connection to Supabase fails
        """
        try:
            # Check and load environment variables
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if not os.path.exists(env_path):
                raise FileNotFoundError(f".env file not found at {env_path}")
            
            load_dotenv(dotenv_path=env_path)
            
            # Get and validate credentials
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("Missing Supabase credentials in .env file")
            if "your_supabase" in supabase_url or "your_supabase" in supabase_key:
                raise ValueError("Please replace placeholder values in .env with actual credentials")
            
            # Initialize client
            client = create_client(supabase_url, supabase_key)
            
            # Test connection
            SupabaseManager._test_connection(client)
            
            logger.info("Supabase client initialized successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise ConnectionError(f"Supabase initialization failed: {str(e)}")
    
    @staticmethod
    def _test_connection(client: Client) -> None:
        """
        Test the Supabase connection by making a simple query.
        
        Args:
            client: Initialized Supabase client
            
        Raises:
            ConnectionError: If connection test fails
        """
        try:
            # Make a simple query to test the connection
            result = client.table('exams').select('count', count='exact').execute()
            if not isinstance(result, dict) and not hasattr(result, 'count'):
                raise ConnectionError("Invalid response format from Supabase")
            logger.debug("Supabase connection test successful")
            
        except Exception as e:
            logger.error(f"Supabase connection test failed: {str(e)}")
            raise ConnectionError(f"Could not connect to Supabase: {str(e)}")


class DatabaseManager:
    def create_notification(self, notification_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a new notification record to the notifications table."""
        try:
            result = self._safe_query(
                self.supabase.table("notifications").insert(notification_data)
            )
            return result.get('data', [None])[0]
        except Exception as e:
            logger.error(f"Error adding notification: {str(e)}")
            return None
    """Handles all database operations with proper error handling"""
    
    def __init__(self, supabase_client: Client):
        """
        Initialize the DatabaseManager with a Supabase client.
        
        Args:
            supabase_client: Initialized Supabase client
            
        Raises:
            ValueError: If supabase_client is None
        """
        if not supabase_client:
            raise ValueError("Supabase client cannot be None")
        self.supabase = supabase_client
    
    def _safe_query(self, query) -> Dict[str, Any]:
        """
        Execute a query with proper error handling
        
        Args:
            query: Supabase query object
            
        Returns:
            Dictionary containing response data or error information
        """
        try:
            response = query.execute()
            
            # Handle response object
            if hasattr(response, 'data'):
                return {'data': response.data}
            # Handle dictionary response
            elif isinstance(response, dict) and 'data' in response:
                return response
            # Handle other response types
            return {'data': response}
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return {'error': str(e), 'data': []}
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the dashboard"""
        try:
            exams = self._safe_query(
                self.supabase.table("exams").select("*")
            )
            alerts = self._safe_query(
                self.supabase.table("alerts").select("*")
            )
            
            exam_data = exams.get('data', [])
            alert_data = alerts.get('data', [])
            
            return {
                'total_exams': len(exam_data),
                'active_exams': len([e for e in exam_data if e.get('status') == 'running']),
                'completed_exams': len([e for e in exam_data if e.get('status') == 'completed']),
                'total_alerts': len(alert_data),
                'pending_reviews': len([a for a in alert_data if not a.get('reviewed', False)]),
                'recent_alerts': sorted(
                    [a for a in alert_data if not a.get('reviewed', False)],
                    key=lambda x: x.get('timestamp', ''),
                    reverse=True
                )[:5],
                'upcoming_exams': sorted(
                    [e for e in exam_data if e.get('status') == 'scheduled'],
                    key=lambda x: x.get('start_time', '')
                )[:5]
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {str(e)}")
            return {
                'error': str(e),
                'total_exams': 0,
                'active_exams': 0,
                'completed_exams': 0,
                'total_alerts': 0,
                'pending_reviews': 0,
                'recent_alerts': [],
                'upcoming_exams': []
            }
    
    def get_exam(self, exam_id: str) -> Optional[Dict[str, Any]]:
        """Get an exam by its ID"""
        try:
            result = self._safe_query(
                self.supabase.table("exams")
                .select("*")
                .eq("id", exam_id)
            )
            return result.get('data', [None])[0]
        except Exception as e:
            logger.error(f"Error getting exam {exam_id}: {str(e)}")
            return None
    
    def get_exams_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all exams with a specific status"""
        try:
            result = self._safe_query(
                self.supabase.table("exams")
                .select("*")
                .eq("status", status)
                .order("start_time", desc=False)
            )
            return result.get('data', [])
        except Exception as e:
            logger.error(f"Error getting exams by status {status}: {str(e)}")
            return []
    
    def get_exam_detections(self, exam_id: str) -> List[Dict[str, Any]]:
        """Get all detections for a specific exam"""
        try:
            result = self._safe_query(
                self.supabase.table("detections")
                .select("*")
                .eq("exam_id", exam_id)
                .order("timestamp", desc=False)
            )
            return result.get('data', [])
        except Exception as e:
            logger.error(f"Error getting detections for exam {exam_id}: {str(e)}")
            return []
    
    def get_exam_alerts(self, exam_id: str) -> List[Dict[str, Any]]:
        """Get all alerts for a specific exam"""
        try:
            result = self._safe_query(
                self.supabase.table("alerts")
                .select("*")
                .eq("exam_id", exam_id)
                .order("timestamp", desc=False)
            )
            return result.get('data', [])
        except Exception as e:
            logger.error(f"Error getting alerts for exam {exam_id}: {str(e)}")
            return []
    
    def update_exam(self, exam_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update exam data"""
        try:
            result = self._safe_query(
                self.supabase.table("exams")
                .update(updates)
                .eq("id", exam_id)
            )
            return result.get('data', [None])[0]
        except Exception as e:
            logger.error(f"Error updating exam {exam_id}: {str(e)}")
            return None
    
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update alert data"""
        try:
            result = self._safe_query(
                self.supabase.table("alerts")
                .update(updates)
                .eq("id", alert_id)
            )
            return result.get('data', [None])[0]
        except Exception as e:
            logger.error(f"Error updating alert {alert_id}: {str(e)}")
            return None
    
    def add_detection(self, detection_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a new detection record"""
        try:
            result = self._safe_query(
                self.supabase.table("detections")
                .insert(detection_data)
            )
            return result.get('data', [None])[0]
        except Exception as e:
            logger.error(f"Error adding detection: {str(e)}")
            return None
    
    def add_alert(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a new alert record"""
        try:
            result = self._safe_query(
                self.supabase.table("alerts")
                .insert(alert_data)
            )
            return result.get('data', [None])[0]
        except Exception as e:
            logger.error(f"Error adding alert: {str(e)}")
            return None

# Initialize the database connection when module is imported
try:
    SUPABASE_CLIENT = SupabaseManager.initialize_client()
    DB_MANAGER = DatabaseManager(SUPABASE_CLIENT)
    logger.info("Database module initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize database module: {str(e)}")
    SUPABASE_CLIENT = None
    DB_MANAGER = None
