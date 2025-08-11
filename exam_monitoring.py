import logging
import threading
import queue
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, List
import pandas as pd
import pytz
import streamlit as st

logger = logging.getLogger(__name__)

class ExamMonitoringSystem:
    def __init__(self, db_manager, video_processor, alert_manager):
        self.db_manager = db_manager
        self.video_processor = video_processor
        self.alert_manager = alert_manager
        self._frame_queues: Dict[str, queue.Queue] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._processing_threads: Dict[str, threading.Thread] = {}
        self._exam_cache: Dict[str, Dict] = {}

    # ====== Core Backend ======
    def start_exam_tracking(self, exam_id: str) -> bool:
        if exam_id in self._processing_threads:
            logger.warning("Exam tracking already running for %s", exam_id)
            return True
        exam = self.get_active_exam(exam_id)
        if not exam:
            logger.error("Cannot start tracking: exam not found %s", exam_id)
            return False
        q = queue.Queue(maxsize=20)
        ev = threading.Event()
        t = threading.Thread(target=self._processing_loop, args=(exam_id, q, ev, exam), daemon=True)
        self._frame_queues[exam_id] = q
        self._stop_events[exam_id] = ev
        self._processing_threads[exam_id] = t
        self._exam_cache[exam_id] = exam
        t.start()
        logger.info("Started exam tracking for %s", exam_id)
        return True

    def stop_exam_tracking(self, exam_id: str) -> bool:
        if exam_id not in self._stop_events:
            return False
        self._stop_events[exam_id].set()
        thread = self._processing_threads.get(exam_id)
        if thread:
            thread.join(timeout=5.0)
        self._frame_queues.pop(exam_id, None)
        self._processing_threads.pop(exam_id, None)
        self._stop_events.pop(exam_id, None)
        try:
            metrics = self.calculate_exam_metrics(exam_id)
            self.db_manager.supabase.table("exam_metrics").insert(metrics).execute()
            logger.info("Saved metrics for %s", exam_id)
        except Exception as e:
            logger.exception("Error saving metrics: %s", e)
        self._exam_cache.pop(exam_id, None)
        return True

    def attach_frame(self, exam_id: str, frame) -> bool:
        q = self._frame_queues.get(exam_id)
        if not q:
            return False
        try:
            q.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def _processing_loop(self, exam_id: str, q: queue.Queue, stop_ev: threading.Event, exam_meta: Dict):
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
            settings = exam_meta.get("monitoring_settings", {}) or {}
            confidence_threshold = settings.get("confidence_threshold", 0.5)
            for det in detections or []:
                conf = float(getattr(det, "confidence", 0.0))
                if conf < confidence_threshold:
                    continue
                row = {
                    "exam_id": exam_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detection_type": getattr(det, "type", "unknown"),
                    "confidence": conf,
                    "people_count": getattr(det, "people_count", None),
                    "details": det.to_dict() if hasattr(det, "to_dict") else {}
                }
                self.db_manager.supabase.table("detections").insert(row).execute()
                if getattr(det, "is_face", False) and settings.get("face_detection", True):
                    self._handle_face_detection(exam_id, frame, det)
                elif getattr(det, "is_object", False) and settings.get("object_detection", True):
                    self._handle_object_detection(exam_id, frame, det)
                elif getattr(det, "is_motion", False) and settings.get("motion_detection", True):
                    self._handle_motion_detection(exam_id, frame, det)

    def get_active_exam(self, exam_id: str) -> Optional[Dict]:
        try:
            scheduled = self.db_manager.supabase.table("scheduled_exams").select("*").eq("id", exam_id).single().execute()
            if not scheduled.data:
                return None
            config = self.db_manager.supabase.table("exams").select("*").eq("id", scheduled.data["exam_id"]).single().execute()
            if not config.data:
                return None
            return {**config.data, **scheduled.data}
        except Exception:
            logger.exception("get_active_exam failed")
            return None

    def calculate_exam_metrics(self, exam_id: str) -> Dict:
        try:
            result = self.db_manager.supabase.table("detections").select("*").eq("exam_id", exam_id).execute()
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
                # Times / durations: try to get from scheduled_exams
                "monitoring_duration": None,
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

    def _handle_face_detection(self, exam_id, frame, det): pass
    def _handle_object_detection(self, exam_id, frame, det): pass
    def _handle_motion_detection(self, exam_id, frame, det): pass

    # ====== UI Rendering ======
    def render_monitoring_interface(self):
        now = datetime.now(pytz.UTC)
        scheduled = self.db_manager.supabase.table("scheduled_exams").select("*").gte("end_time", now.isoformat()).execute()
        if not scheduled.data:
            st.info("No active or upcoming exams found")
            return
        config_ids = [e['exam_id'] for e in scheduled.data]
        configs = self.db_manager.supabase.table("exams").select("*").in_("id", config_ids).execute()
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

        st.header("🟢 Current Exams")
        for exam in current_exams:
            st.subheader(f"{exam['exam_name']} - {exam['course_code']}")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Venue: {exam['venue']}")
                st.write(f"Instructor: {exam['instructor']}")
            with col2:
                st.write(f"Students: {exam['total_students']}")
            is_running = exam['id'] in self._processing_threads
            if not is_running:
                if st.button(f"▶️ Start Monitoring {exam['exam_name']}", key=f"start_{exam['id']}"):
                    self.start_exam_tracking(exam['id'])
                    st.success(f"Started monitoring {exam['exam_name']}")
            else:
                if st.button(f"⏹ Stop Monitoring {exam['exam_name']}", key=f"stop_{exam['id']}"):
                    self.stop_exam_tracking(exam['id'])
                    st.success(f"Stopped monitoring {exam['exam_name']} and saved metrics")
            if st.button("📊 View Metrics", key=f"metrics_{exam['id']}"):
                metrics = self.db_manager.supabase.table("exam_metrics").select("*").eq("exam_id", exam['id']).order("created_at", desc=True).limit(1).execute()
                if metrics.data:
                    st.dataframe(pd.DataFrame([metrics.data[0]]))
                else:
                    st.info("No metrics found")

        st.header("📅 Upcoming Exams")
        if upcoming_exams:
            st.dataframe(pd.DataFrame(upcoming_exams))
        else:
            st.info("No upcoming exams")


