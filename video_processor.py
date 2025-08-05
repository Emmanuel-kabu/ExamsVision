import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class Detection:
    def __init__(self, type_name: str, confidence: float, box: Optional[Tuple[int, int, int, int]] = None):
        self.type_name = type_name
        self.confidence = confidence
        self.box = box
        self.is_face = type_name == 'face'
        self.is_object = type_name == 'object'
        self.is_motion = type_name == 'motion'

class VideoProcessor:
    """Handles video processing and detection tasks"""
    
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        self.frame_count = 0
        self.previous_frame = None
        
    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """Process a frame and return all detections"""
        detections = []
        self.frame_count += 1
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Add face detections
            for (x, y, w, h) in faces:
                detections.append(Detection(
                    'face',
                    confidence=0.8,  # Haar cascade doesn't provide confidence
                    box=(x, y, w, h)
                ))
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Motion detection
            if self.frame_count % 3 == 0:  # Process every 3rd frame for motion
                fgmask = self.bg_subtractor.apply(frame)
                _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Filter small movements
                        (x, y, w, h) = cv2.boundingRect(contour)
                        detections.append(Detection(
                            'motion',
                            confidence=0.7,
                            box=(x, y, w, h)
                        ))
                        # Draw rectangle around motion
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Update previous frame
            self.previous_frame = gray
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []
    
    def add_detections_to_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Add detection visualizations to frame"""
        try:
            for detection in detections:
                if detection.box:
                    x, y, w, h = detection.box
                    if detection.is_face:
                        color = (0, 255, 0)  # Green for faces
                    elif detection.is_motion:
                        color = (0, 0, 255)  # Red for motion
                    else:
                        color = (255, 0, 0)  # Blue for other objects
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(
                        frame,
                        f"{detection.type_name} ({detection.confidence:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding detections to frame: {e}")
            return frame
    
    def release(self):
        """Release any resources"""
        pass  # Add cleanup code if needed
