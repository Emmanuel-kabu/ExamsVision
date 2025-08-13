import logging
import threading
import queue
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
import pytz

logger = logging.getLogger(__name__)

class ExamMonitoringSystem:
    def __init__(self, db_manager, video_processor, alert_manager):
        self.db_manager = db_manager
        self.video_processor = video_processor
        self.alert_manager = alert_manager

        # Internal tracking
        self._frame_queues: Dict[str, queue.Queue] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._processing_threads: Dict[str, threading.Thread] = {}
        self._exam_cache: Dict[str, Dict] = {}

    # --------------------------
    # Public API
    # --------------------------
    def start_exam_tracking(self, exam_id: str) -> bool:
        """Start processing frames for a given exam."""
        if exam_id in self._processing_threads:
            logger.warning("Exam tracking already running for %s", exam_id)
            return True

        exam = self.get_active_exam(exam_id)
        if not exam:
            logger.error("Cannot start tracking: exam not found %s", exam_id)
            return False

        q = queue.Queue(maxsize=20)
        ev = threading.Event()
        t = threading.Thread(
            target=self._processing_loop,
            args=(exam_id, q, ev, exam),
            daemon=True
        )

        self._frame_queues[exam_id] = q
        self._stop_events[exam_id] = ev
        self._processing_threads[exam_id] = t
        self._exam_cache[exam_id] = exam

        t.start()
        logger.info("Started exam tracking for %s", exam_id)
        return True

    def stop_exam_tracking(self, exam_id: str) -> bool:
        """Stop processing frames for a given exam and save metrics."""
        if exam_id not in self._stop_events:
            return False

        self._stop_events[exam_id].set()

        thread = self._processing_threads.get(exam_id)
        if thread:
            thread.join(timeout=5.0)

        # Cleanup
        self._frame_queues.pop(exam_id, None)
        self._processing_threads.pop(exam_id, None)
        self._stop_events.pop(exam_id, None)
        self._exam_cache.pop(exam_id, None)

        # Save metrics
        try:
            metrics = self.calculate_exam_metrics(exam_id)
            if metrics:
                self.db_manager.supabase.table("exam_metrics").insert(metrics).execute()
                logger.info("Saved metrics for %s", exam_id)
        except Exception as e:
            logger.exception("Error saving metrics: %s", e)

        return True

    def attach_frame(self, exam_id: str, frame) -> bool:
        """Attach a frame to the queue for this exam."""
        q = self._frame_queues.get(exam_id)
        if not q:
            return False
        try:
            q.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def list_exams(self) -> Tuple[List[Dict], List[Dict]]:
        """Return current and upcoming exams."""
        now = datetime.now(pytz.UTC)
        scheduled = self.db_manager.supabase.table("scheduled_exams") \
            .select("*") \
            .gte("end_time", now.isoformat()) \
            .execute()

        if not scheduled.data:
            return [], []

        config_ids = [e['exam_id'] for e in scheduled.data]
        configs = self.db_manager.supabase.table("exams") \
            .select("*") \
            .in_("id", config_ids) \
            .execute()

        config_map = {c['id']: c for c in configs.data}
        current_exams, upcoming_exams = [], []

        for exam in scheduled.data:
            cfg = config_map.get(exam['exam_id'])
            if not cfg:
                continue
            merged = {**cfg, **exam}
            start_time = datetime.fromisoformat(exam['start_time'])
            end_time = datetime.fromisoformat(exam['end_time'])
            if start_time <= now <= end_time:
                current_exams.append(merged)
            elif start_time > now:
                upcoming_exams.append(merged)

        return current_exams, upcoming_exams

    # --------------------------
    # Internal processing
    # --------------------------
    def _processing_loop(self, exam_id: str, q: queue.Queue, stop_ev: threading.Event, exam_meta: Dict):
        """Consume frames from queue and process detections."""
        settings = exam_meta.get("monitoring_settings", {}) or {}
        confidence_threshold = settings.get("confidence_threshold", 0.5)

        while not stop_ev.is_set():
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                detections = self.video_processor.process_frame(frame)
            except Exception as e:
                logger.exception("Video processor failed: %s", e)
                continue

            for det in detections or []:
                conf = float(getattr(det, "confidence", 0.0))
                if conf < confidence_threshold:
                    continue

                # Insert detection into DB
                row = {
                    "exam_id": exam_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detection_type": getattr(det, "type", "unknown"),
                    "confidence": conf,
                    "people_count": getattr(det, "people_count", None),
                    "details": det.to_dict() if hasattr(det, "to_dict") else {}
                }
                self.db_manager.supabase.table("detections").insert(row).execute()

    def get_active_exam(self, exam_id: str) -> Optional[Dict]:
        """Fetch exam config and schedule details."""
        try:
            scheduled = self.db_manager.supabase.table("scheduled_exams") \
                .select("*").eq("id", exam_id).single().execute()
            if not scheduled.data:
                return None

            config = self.db_manager.supabase.table("exams") \
                .select("*").eq("id", scheduled.data["exam_id"]).single().execute()
            if not config.data:
                return None

            return {**config.data, **scheduled.data}
        except Exception:
            logger.exception("get_active_exam failed")
            return None

    def calculate_exam_metrics(self, exam_id: str) -> Dict:
        """Calculate summary statistics for detections."""
        try:
            result = self.db_manager.supabase.table("detections") \
                .select("*").eq("exam_id", exam_id).execute()
            detections = result.data or []
            total_detections = len(detections)
            cheating_count = sum(1 for d in detections if d["detection_type"] != "good_behavior")
            confidences = [d["confidence"] for d in detections if d["confidence"] is not None]

            return {
                "exam_id": exam_id,
                "total_detections": total_detections,
                "total_cheating": cheating_count,
                "total_good_behavior": total_detections - cheating_count,
                "cheating_percentage": (cheating_count / total_detections) * 100 if total_detections else 0,
                "good_behavior_percentage": ((total_detections - cheating_count) / total_detections) * 100 if total_detections else 0,
                "mean_confidence": np.mean(confidences) if confidences else None,
                "median_confidence": float(np.median(confidences)) if confidences else None,
                "max_confidence": float(max(confidences)) if confidences else None,
                "min_confidence": float(min(confidences)) if confidences else None,
                "std_deviation": float(np.std(confidences)) if confidences else None,
                "metrics_data": {"counts_by_type": self._group_by_type(detections)}
            }
        except Exception:
            logger.exception("calculate_exam_metrics failed")
            return {}

    def _group_by_type(self, detections):
        counts = {}
        for d in detections:
            t = d.get("detection_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts