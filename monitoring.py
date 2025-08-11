import logging
import threading
import queue
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, List

import pandas as pd  # used only for UI display in render_monitoring_interface if needed

logger = logging.getLogger(__name__)

class ExamMonitoringSystem:
    """Exam tracking + detection processing (NO camera control)."""

    def __init__(self, db_manager, video_processor, alert_manager):
        self.db_manager = db_manager
        self.video_processor = video_processor
        self.alert_manager = alert_manager

        # Per-exam runtime state
        self._frame_queues: Dict[str, queue.Queue] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._processing_threads: Dict[str, threading.Thread] = {}

        # Small in-memory cache for exam meta (optional)
        self._exam_cache: Dict[str, Dict] = {}

    # -------------------------
    # Public API (main app calls these)
    # -------------------------
    def start_exam_tracking(self, exam_id: str) -> bool:
        """
        Start tracking for an exam.
        NOTE: this does NOT open camera. Main app must feed frames via attach_frame().
        """
        if exam_id in self._processing_threads:
            logger.warning("Exam tracking already running for %s", exam_id)
            return True

        exam = self.get_active_exam(exam_id)
        if not exam:
            logger.error("Cannot start tracking: exam not found %s", exam_id)
            return False

        # create resources
        q = queue.Queue(maxsize=20)
        ev = threading.Event()
        t = threading.Thread(target=self._processing_loop, args=(exam_id, q, ev, exam), daemon=True)

        self._frame_queues[exam_id] = q
        self._stop_events[exam_id] = ev
        self._processing_threads[exam_id] = t
        self._exam_cache[exam_id] = exam

        t.start()
        logger.info("Started exam tracking for %s (%s)", exam_id, exam.get("exam_name"))
        return True

    def stop_exam_tracking(self, exam_id: str) -> bool:
        """
        Stop exam tracking, wait for thread to finish, calculate & save metrics.
        Call this after your main app has stopped the camera.
        """
        if exam_id not in self._stop_events:
            logger.warning("stop_exam_tracking called but tracking not running for %s", exam_id)
            return False

        # Signal processing thread to stop
        self._stop_events[exam_id].set()

        # Wait for thread to finish gracefully
        thread = self._processing_threads.get(exam_id)
        if thread:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Processing thread for %s did not stop in time", exam_id)

        # clean up runtime resources
        self._frame_queues.pop(exam_id, None)
        self._processing_threads.pop(exam_id, None)
        self._stop_events.pop(exam_id, None)

        # Compute & save metrics
        try:
            metrics = self.calculate_exam_metrics(exam_id)
            # Ensure required fields are present for insert
            metrics_to_insert = metrics.copy()
            # Insert into exam_metrics
            res = self.db_manager.supabase.table("exam_metrics").insert(metrics_to_insert).execute()
            if getattr(res, "error", None):
                logger.error("Failed to save metrics for %s: %s", exam_id, res.error)
            else:
                logger.info("Saved metrics for exam %s", exam_id)
        except Exception as e:
            logger.exception("Error saving metrics for %s: %s", exam_id, e)
            return False

        # Optionally remove exam cache
        self._exam_cache.pop(exam_id, None)
        return True

    def attach_frame(self, exam_id: str, frame) -> bool:
        """
        Called by main app for each captured frame. Places frame into processing queue.
        Return True if queued, False otherwise (queue full or not running).
        """
        q = self._frame_queues.get(exam_id)
        if not q:
            return False
        try:
            q.put_nowait(frame)
            return True
        except queue.Full:
            # drop frame if processing can't keep up
            return False

    # -------------------------
    # Internal processing loop
    # -------------------------
    def _processing_loop(self, exam_id: str, q: queue.Queue, stop_ev: threading.Event, exam_meta: Dict):
        """
        Pull frames from queue, run detection, save detections & trigger alerts.
        This loop runs in a background thread spawned by start_exam_tracking().
        """
        logger.debug("Processing loop started for %s", exam_id)

        # Optional runtime accumulators for metrics (kept in memory)
        detection_records = []  # small cache of dicts to optionally bulk-insert later
        first_frame_ts = None
        last_frame_ts = None

        try:
            while not stop_ev.is_set():
                try:
                    frame = q.get(timeout=0.5)  # wait for frame
                except queue.Empty:
                    continue

                # mark timestamps
                now_ts = datetime.now(timezone.utc)
                if first_frame_ts is None:
                    first_frame_ts = now_ts
                last_frame_ts = now_ts

                # Run your model / video_processor
                try:
                    detections = self.video_processor.process_frame(frame)
                except Exception as e:
                    logger.exception("Video processor failed for %s: %s", exam_id, e)
                    continue

                # check settings in exam_meta (fallbacks)
                settings = exam_meta.get("monitoring_settings", {}) or {}
                confidence_threshold = settings.get("confidence_threshold", 0.5)

                # iterate detections and insert rows
                for det in detections or []:
                    try:
                        conf = float(getattr(det, "confidence", getattr(det, "conf", 0.0)))
                    except Exception:
                        conf = 0.0

                    if conf < confidence_threshold:
                        continue

                    # Build detection row for DB
                    detection_row = {
                        "exam_id": exam_id,
                        "timestamp": now_ts.isoformat(),
                        "detection_type": getattr(det, "type", getattr(det, "label", "unknown")),
                        "confidence": float(conf),
                        "people_count": getattr(det, "people_count", None),
                        "details": det.to_dict() if hasattr(det, "to_dict") else {}
                    }

                    # Insert immediately (safe, keeps memory small)
                    try:
                        res = self.db_manager.supabase.table("detections").insert(detection_row).execute()
                        if getattr(res, "error", None):
                            logger.error("Failed to insert detection for %s: %s", exam_id, res.error)
                        else:
                            logger.debug("Inserted detection for %s", exam_id)
                    except Exception:
                        logger.exception("DB insert for detection failed for exam %s", exam_id)

                    # trigger alerts (separate logic — unchanged)
                    try:
                        # call alert manager — it will save evidence etc.
                        if getattr(det, "is_face", False) and settings.get("face_detection", True):
                            self._handle_face_detection(exam_id, frame, det)
                        elif getattr(det, "is_object", False) and settings.get("object_detection", True):
                            self._handle_object_detection(exam_id, frame, det)
                        elif getattr(det, "is_motion", False) and settings.get("motion_detection", True):
                            self._handle_motion_detection(exam_id, frame, det)
                    except Exception:
                        logger.exception("Alert handling failed for exam %s", exam_id)

            # end while
        except Exception:
            logger.exception("Unexpected error in processing loop for %s", exam_id)
        finally:
            # optionally flush any remaining caches (not required; detections were inserted inline)
            logger.debug("Processing loop exiting for %s (frames processed between %s and %s)",
                         exam_id, first_frame_ts, last_frame_ts)

    # -------------------------
    # DB helper: get active exam (combine scheduled_exams + exams)
    # -------------------------
    def get_active_exam(self, exam_id: str) -> Optional[Dict]:
        try:
            scheduled = self.db_manager.supabase.table("scheduled_exams").select("*").eq("id", exam_id).single().execute()
            if not scheduled or not getattr(scheduled, "data", None):
                return None
            scheduled_data = scheduled.data

            # fetch base exam config
            exam_cfg = self.db_manager.supabase.table("exams").select("*").eq("id", scheduled_data["exam_id"]).single().execute()
            if not exam_cfg or not getattr(exam_cfg, "data", None):
                return None
            cfg = exam_cfg.data

            # merge: scheduled takes precedence
            merged = {**cfg, **scheduled_data}
            return merged
        except Exception:
            logger.exception("get_active_exam failed for %s", exam_id)
            return None

    # -------------------------
    # Metrics calculation & helpers
    # -------------------------
    def calculate_exam_metrics(self, exam_id: str) -> Dict:
        """
        Build an exam_metrics dict to insert into exam_metrics table.
        Uses rows from detections table for the exam.
        """
        try:
            resp = self.db_manager.supabase.table("detections").select("*").eq("exam_id", exam_id).execute()
            rows = resp.data or []

            total_detections = len(rows)
            confidences = [r.get("confidence") for r in rows if r.get("confidence") is not None]
            # Example classification: treat anything not labeled 'good_behavior' as cheating
            cheating_count = sum(1 for r in rows if r.get("detection_type") and r.get("detection_type") != "good_behavior")
            good_behavior = total_detections - cheating_count

            metrics = {
                "exam_id": exam_id,
                "total_detections": total_detections,
                "total_cheating": cheating_count,
                "total_good_behavior": good_behavior,
                "cheating_percentage": (cheating_count / total_detections * 100) if total_detections else 0.0,
                "good_behavior_percentage": (good_behavior / total_detections * 100) if total_detections else 100.0,
                "mean_confidence": float(np.mean(confidences)) if confidences else None,
                "median_confidence": float(np.median(confidences)) if confidences else None,
                "max_confidence": float(max(confidences)) if confidences else None,
                "min_confidence": float(min(confidences)) if confidences else None,
                "std_deviation": float(np.std(confidences)) if confidences else None,
                # Times / durations: try to get from scheduled_exams
                "monitoring_duration": None,
                "metrics_data": {"counts_by_type": self._group_by_type(rows)}
            }

            # attempt to compute monitoring_duration using scheduled times if they exist
            try:
                scheduled = self.db_manager.supabase.table("scheduled_exams").select("*").eq("id", exam_id).single().execute()
                if getattr(scheduled, "data", None):
                    s = scheduled.data
                    if s.get("start_time") and s.get("end_time"):
                        # convert from ISO string if necessary:
                        start = s["start_time"]
                        end = s["end_time"]
                        if isinstance(start, str):
                            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        if isinstance(end, str):
                            end = datetime.fromisoformat(end.replace("Z", "+00:00"))
                        metrics["monitoring_duration"] = (end - start).total_seconds()
            except Exception:
                logger.debug("Could not compute monitoring_duration from scheduled_exams for %s", exam_id)

            return metrics
        except Exception:
            logger.exception("calculate_exam_metrics failed for %s", exam_id)
            return {
                "exam_id": exam_id,
                "total_detections": 0,
                "total_cheating": 0,
                "total_good_behavior": 0,
                "cheating_percentage": 0.0,
                "good_behavior_percentage": 100.0,
                "metrics_data": {}
            }

    def _group_by_type(self, detections_rows: List[Dict]) -> Dict[str, int]:
        counts = {}
        for r in detections_rows:
            k = r.get("detection_type", "unknown")
            counts[k] = counts.get(k, 0) + 1
        return counts

    # Alert handlers remain thin wrappers
    def _handle_face_detection(self, exam_id, frame, detection):
        try:
            self.alert_manager.create_alert(exam_id=exam_id, alert_type="face_detection", confidence=getattr(detection, "confidence", 0.0), frame=frame)
        except Exception:
            logger.exception("Face alert error for %s", exam_id)

    def _handle_object_detection(self, exam_id, frame, detection):
        try:
            self.alert_manager.create_alert(exam_id=exam_id, alert_type="object_detection", confidence=getattr(detection, "confidence", 0.0), frame=frame)
        except Exception:
            logger.exception("Object alert error for %s", exam_id)

    def _handle_motion_detection(self, exam_id, frame, detection):
        try:
            self.alert_manager.create_alert(exam_id=exam_id, alert_type="motion_detection", confidence=getattr(detection, "confidence", 0.0), frame=frame)
        except Exception:
            logger.exception("Motion alert error for %s", exam_id)