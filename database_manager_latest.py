import logging
import os
import time
from typing import List, Dict, Optional
from datetime import datetime
from supabase import Client, create_client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_supabase() -> Client:
    """Initialize Supabase client using environment variables."""
    try:
        # Check for .env file in multiple locations
        env_paths = [
            os.path.join(os.path.dirname(__file__), '.env'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        ]
        
        env_file_found = False
        for env_path in env_paths:
            if os.path.exists(env_path):
                # Load environment variables from the found path
                load_dotenv(dotenv_path=env_path)
                env_file_found = True
                logger.info(f"Loaded .env file from {env_path}")
                break
                
        if not env_file_found:
            logger.error("No .env file found in searched locations")
            raise ValueError("Missing .env file. Create one with SUPABASE_URL and SUPABASE_KEY")
            
        # Get Supabase credentials from environment
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        # Validate credentials
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        
        if supabase_url == "your_supabase_project_url" or supabase_key == "your_supabase_anon_key":
            raise ValueError("Please replace the placeholder values in .env with your actual Supabase credentials")
        
        # Initialize Supabase client with timeout
        try:
            supabase: Client = create_client(
                supabase_url,
                supabase_key,
                options={'timeout': 10}  # 10 second timeout
            )
            
            # Test connection with retry logic
            max_retries = 3
            retry_delay = 2
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Simple health check query
                    response = supabase.from_('exams').select('count').limit(1).execute()
                    
                    # Verify response format
                    if isinstance(response, dict):
                        if 'error' in response:
                            raise ValueError(f"Database error: {response['error']['message']}")
                        logger.info("✓ Supabase client initialized and connected successfully")
                        return supabase
                    else:
                        logger.info("✓ Supabase client initialized and connected successfully")
                        return supabase
                        
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection attempt {attempt + 1} failed: {last_error}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    
            logger.error(f"Failed to connect after {max_retries} attempts. Last error: {last_error}")
            raise ConnectionError(f"Could not connect to Supabase: {last_error}")
                
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {str(e)}")
            raise ConnectionError(f"Could not initialize Supabase client: {str(e)}")
            
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {str(e)}")
        raise

# Initialize the global Supabase client
try:
    logger.info("Initializing global Supabase client...")
    SUPABASE_CLIENT = init_supabase()
except Exception as e:
    logger.error(f"Failed to initialize global Supabase client: {str(e)}")
    SUPABASE_CLIENT = None

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, supabase: Client):
        """Initialize with Supabase client"""
        self.supabase = supabase
        
    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for the dashboard"""
        try:
            # Get exam counts by status
            exams_response = self.supabase.from_('exams').select('*').execute()
            exam_data = exams_response['data'] if isinstance(exams_response, dict) and 'data' in exams_response else []
            
            # Get alerts
            alerts_response = self.supabase.from_('alerts').select('*').execute()
            alert_data = alerts_response['data'] if isinstance(alerts_response, dict) and 'data' in alerts_response else []
            
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
            result = self.supabase.from_('exams').select('*').eq('id', exam_id).execute()
            if isinstance(result, dict) and 'data' in result and result['data']:
                return result['data'][0]
            return None
        except Exception as e:
            logger.error(f"Error getting exam: {e}")
            raise
