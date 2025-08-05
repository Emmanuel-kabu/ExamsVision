import time
from datetime import datetime
import threading
import logging
from typing import Dict, Any
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyncManager:
    """Manages data synchronization with Supabase"""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.detection_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.is_syncing = False
        self.sync_thread = None
        self._setup_sync_thread()
    
    def _setup_sync_thread(self):
        """Initialize the background sync thread"""
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.is_syncing = True
        self.sync_thread.start()
        logger.info("Sync thread started")
    
    def queue_detection(self, detection_data: Dict[str, Any]):
        """Queue a detection for syncing"""
        self.detection_queue.put(detection_data)
    
    def queue_alert(self, alert_data: Dict[str, Any]):
        """Queue an alert for syncing"""
        self.alert_queue.put(alert_data)
    
    def _sync_loop(self):
        """Main synchronization loop"""
        while self.is_syncing:
            try:
                # Process detections
                while not self.detection_queue.empty():
                    detection = self.detection_queue.get()
                    self._sync_detection(detection)
                    self.detection_queue.task_done()
                
                # Process alerts
                while not self.alert_queue.empty():
                    alert = self.alert_queue.get()
                    self._sync_alert(alert)
                    self.alert_queue.task_done()
                
                time.sleep(1)  # Prevent busy waiting
            except Exception as e:
                logger.error(f"Sync error: {str(e)}")
    
    def _sync_detection(self, detection: Dict[str, Any]):
        """Sync a single detection to Supabase"""
        try:
            self.supabase.table("detections").insert(detection).execute()
            logger.info(f"Detection synced: {detection.get('id')}")
        except Exception as e:
            logger.error(f"Failed to sync detection: {str(e)}")
    
    def _sync_alert(self, alert: Dict[str, Any]):
        """Sync a single alert to Supabase"""
        try:
            self.supabase.table("alerts").insert(alert).execute()
            logger.info(f"Alert synced: {alert.get('id')}")
        except Exception as e:
            logger.error(f"Failed to sync alert: {str(e)}")
    
    def stop_sync(self):
        """Stop the sync thread"""
        self.is_syncing = False
        if self.sync_thread:
            self.sync_thread.join()
            logger.info("Sync thread stopped")
