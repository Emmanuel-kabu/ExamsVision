"""Configuration settings for the application"""

import os
from datetime import timedelta

# Alert configuration
ALERT_CONFIG = {
    "save_images": True,
    "alert_cooldown": timedelta(seconds=30),  # Minimum time between alerts of the same type
    "confidence_threshold": 0.5,  # Default confidence threshold for detections
}

# Detection settings
DETECTION_CONFIG = {
    "face_detection": {
        "enabled": True,
        "min_size": (30, 30),
        "scale_factor": 1.1,
        "min_neighbors": 5,
    },
    "motion_detection": {
        "enabled": True,
        "history": 500,
        "var_threshold": 16,
        "detect_shadows": True,
        "min_area": 500,  # Minimum area for motion detection
    }
}

# Path configuration
PATH_CONFIG = {
    "detection_images": "cheating_detections",
    "alert_images": "alert_images",
    "logs": "logs"
}

# Ensure directories exist
for path in PATH_CONFIG.values():
    os.makedirs(path, exist_ok=True)

# Database configuration
DB_CONFIG = {
    "alerts_table": "exam_alerts",
    "max_alerts_per_query": 100
}

# Video configuration
VIDEO_CONFIG = {
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30
}

# Monitoring configuration
MONITORING_CONFIG = {
    "max_concurrent_sessions": 4,
    "frame_buffer_size": 10,
    "stats_update_interval": 5  # seconds
}
