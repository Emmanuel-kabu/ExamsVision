import numpy as np
from typing import Dict, Any
from datetime import datetime
import logging

class AnomalyDetector:
    """Detects anomalies in exam monitoring data using statistical methods"""
    
    def __init__(self):
        self.history = []
        self.baseline = {
            'confidence': [],
            'people_count': [],
            'motion_level': [],
            'alert_frequency': []
        }
        self.z_score_threshold = 2.0  # Number of standard deviations for anomaly
        
    def _compute_z_score(self, value: float, history: list) -> float:
        """Compute z-score for a value against historical data"""
        if not history:
            return 0.0
        mean = np.mean(history)
        std = np.std(history) or 1.0  # Avoid division by zero
        return abs((value - mean) / std)
        
    def _update_baseline(self, data: Dict[str, Any]):
        """Update baseline statistics"""
        for key in self.baseline:
            if key in data:
                self.baseline[key].append(data[key])
                # Keep last 100 values for baseline
                self.baseline[key] = self.baseline[key][-100:]
    
    def detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in the current monitoring data
        
        Args:
            data: Dictionary containing current monitoring metrics
                - confidence: Detection confidence
                - people_count: Number of people detected
                - motion_level: Level of motion detected
                - alert_frequency: Frequency of alerts
                - duration: Duration of current behavior
                
        Returns:
            Dictionary containing:
                - is_anomaly: Boolean indicating if anomaly detected
                - score: Anomaly score (0-1)
                - severity: Anomaly severity (low/medium/high)
                - insights: List of specific anomaly insights
        """
        # Update baseline
        self._update_baseline(data)
        
        # Calculate anomaly scores for each metric
        scores = {}
        insights = []
        
        for key in self.baseline:
            if key in data and self.baseline[key]:
                z_score = self._compute_z_score(data[key], self.baseline[key])
                scores[key] = min(1.0, z_score / self.z_score_threshold)
                
                if z_score > self.z_score_threshold:
                    insights.append(f"Unusual {key.replace('_', ' ')}: {data[key]:.2f}")
        
        # Compute overall anomaly score
        if scores:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = 0.0
            
        # Determine severity
        if overall_score > 0.8:
            severity = "high"
        elif overall_score > 0.5:
            severity = "medium"
        else:
            severity = "low"
            
        return {
            'is_anomaly': overall_score > 0.5,
            'score': overall_score,
            'severity': severity,
            'insights': insights
        }
