import pandas as pd
from datetime import datetime, timedelta

class BehaviorAnalyzer:
    """Analyzes behavior patterns from exam monitoring data"""
    
    def __init__(self):
        self.analysis_window = timedelta(minutes=5)  # Default 5-minute window

    def analyze_behavior_patterns(self, detections, motion_level=0.0):
        """Analyze behavior patterns from detection history"""
        if not isinstance(detections, pd.DataFrame) or detections.empty:
            return {
                'alert_level': 'normal',
                'motion_status': 'normal',
                'pattern_detected': False,
                'recommendations': []
            }

        # Get recent detections within analysis window
        now = datetime.now()
        recent_detections = detections[
            detections['timestamp'] >= now - self.analysis_window
        ]

        if recent_detections.empty:
            return {
                'alert_level': 'normal',
                'motion_status': 'normal',
                'pattern_detected': False,
                'recommendations': []
            }

        # Analyze patterns
        total_cheating = recent_detections['cheating'].sum()
        total_good = recent_detections['good'].sum()
        detection_rate = len(recent_detections)
        
        # Determine alert level
        alert_level = 'normal'
        recommendations = []
        
        if total_cheating > total_good:
            alert_level = 'high'
            recommendations.append("High rate of suspicious behavior detected")
        elif total_cheating > 0:
            alert_level = 'medium'
            recommendations.append("Some suspicious behavior detected")

        # Analyze motion
        motion_status = 'normal'
        if motion_level > 0.8:
            motion_status = 'high'
            recommendations.append("Excessive movement detected")
        elif motion_level > 0.5:
            motion_status = 'medium'
            recommendations.append("Increased movement detected")

        # Check for patterns
        pattern_detected = False
        if detection_rate > 10:
            pattern_detected = True
            recommendations.append("High frequency of detections")

        return {
            'alert_level': alert_level,
            'motion_status': motion_status,
            'pattern_detected': pattern_detected,
            'recommendations': recommendations
        }

    def get_historical_patterns(self, detections):
        """Analyze historical patterns across all data"""
        if not isinstance(detections, pd.DataFrame) or detections.empty:
            return {
                'patterns': [],
                'trends': {},
                'summary': "No historical data available"
            }

        # Group by hour to see time-based patterns
        detections['hour'] = detections['timestamp'].dt.hour
        hourly_patterns = detections.groupby('hour').agg({
            'cheating': 'sum',
            'good': 'sum'
        }).reset_index()

        # Find peak hours
        peak_hours = hourly_patterns[
            hourly_patterns['cheating'] == hourly_patterns['cheating'].max()
        ]['hour'].tolist()

        # Calculate trends
        total_detections = len(detections)
        cheating_ratio = detections['cheating'].sum() / max(1, total_detections)

        return {
            'patterns': [
                f"Peak suspicious activity during hour(s): {', '.join(map(str, peak_hours))}"
            ] if peak_hours else [],
            'trends': {
                'total_incidents': int(detections['cheating'].sum()),
                'cheating_ratio': float(cheating_ratio),
                'detection_count': int(total_detections)
            },
            'summary': self._generate_summary(cheating_ratio, peak_hours)
        }

    def _generate_summary(self, cheating_ratio, peak_hours):
        """Generate a human-readable summary of behavior patterns"""
        if not peak_hours:
            return "Insufficient data for pattern analysis"

        severity = "high" if cheating_ratio > 0.5 else "moderate" if cheating_ratio > 0.2 else "low"
        
        return f"Analysis shows {severity} level of suspicious activity" + \
               (f" with peaks during hour(s) {', '.join(map(str, peak_hours))}" if peak_hours else "")
