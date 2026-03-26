# =============================================================================
# VisionTrack Edu — Detector Engine (runs in background thread)
# =============================================================================
# This module owns the webcam + YOLO model.
# The Flask app calls start() / stop() and reads get_frame() + get_stats().
# =============================================================================

import cv2
import sys
import time
import threading
import logging
import numpy as np
from ultralytics import YOLO

from config import (
    WEBCAM_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS,
    FRAME_WIDTH, FRAME_HEIGHT, SKIP_N,
    MODEL_PATH, PERSON_CLASS_ID,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    BOX_COLOR, LABEL_COLOR, JPEG_QUALITY,
    LOG_MIN_PERSONS,
)

log = logging.getLogger("VisionTrack.Detector")

# =============================================================================
# DRAWING HELPERS
# =============================================================================

def _draw_boxes(frame: np.ndarray, boxes: list, confidences: list):
    """Draw bounding boxes + labels on frame (in-place)."""
    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        label = f"Person  {conf:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ly1 = max(y1 - th - bl - 6, 0)
        ly2 = max(y1 - 1, th + bl + 4)

        cv2.rectangle(frame, (x1, ly1), (x1 + tw + 8, ly2), BOX_COLOR, -1)
        cv2.putText(
            frame, label, (x1 + 4, ly2 - bl - 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, LABEL_COLOR, 1, cv2.LINE_AA
        )


def _draw_hud(frame: np.ndarray, fps: float, count: int):
    """Draw a minimal top-left HUD (FPS + count) on the stream frame."""
    lines = [
        f"FPS: {fps:5.1f}",
        f"People: {count}",
    ]
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 230, 100), 1, cv2.LINE_AA)


# =============================================================================
# DETECTOR CLASS
# =============================================================================

class HumanDetector:
    """
    Threaded human detector.

    Usage:
        det = HumanDetector()
        det.start()           # opens webcam, starts inference thread
        frame_bytes = det.get_frame()   # latest JPEG bytes (for MJPEG stream)
        stats       = det.get_stats()   # dict with fps, count, uptime, …
        det.stop()            # releases webcam + model
    """

    def __init__(self):
        self._thread       : threading.Thread | None = None
        self._lock         = threading.Lock()
        self._running      = False

        # Shared state (written by thread, read by Flask)
        self._frame_bytes  : bytes | None = None   # latest JPEG-encoded frame
        self._stats        : dict = {
            "fps"         : 0.0,
            "person_count": 0,
            "total_frames": 0,
            "uptime_sec"  : 0.0,
            "status"      : "stopped",   # stopped | starting | running | error
            "error"       : "",
        }

        self._model        = None
        self._cap          = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Load the YOLO model and open the webcam.
        Launches the inference loop in a daemon thread.
        Returns True on success, False if already running or model fails.
        """
        if self._running:
            log.warning("Detector already running — ignoring start().")
            return False

        self._set_status("starting")
        log.info("Detector starting …")

        # Load model on the calling thread so errors surface immediately
        try:
            log.info(f"Loading {MODEL_PATH} …")
            self._model = YOLO(MODEL_PATH)
            self._model.to("cpu")
            log.info("Model ready ✅")
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            log.error(msg)
            self._set_status("error", error=msg)
            return False

        # Open webcam
        self._cap = self._open_webcam()
        if self._cap is None:
            self._set_status("error", error="Cannot open webcam.")
            return False

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Signal the inference thread to stop and release resources."""
        log.info("Stopping detector …")
        self._running = False
        if self._thread:
            self._thread.join(timeout=4)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._model      = None
        self._frame_bytes = None
        self._set_status("stopped")
        log.info("Detector stopped.")

    def get_frame(self) -> bytes | None:
        """Return the latest JPEG-encoded annotated frame (thread-safe)."""
        with self._lock:
            return self._frame_bytes

    def get_stats(self) -> dict:
        """Return a snapshot of current stats (thread-safe)."""
        with self._lock:
            return dict(self._stats)

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _set_status(self, status: str, **kwargs):
        with self._lock:
            self._stats["status"] = status
            for k, v in kwargs.items():
                self._stats[k] = v

    def _open_webcam(self) -> cv2.VideoCapture | None:
        """Open the webcam with the OS-appropriate backend."""
        if sys.platform == "win32":
            backend, bname = cv2.CAP_DSHOW, "DirectShow"
        elif sys.platform.startswith("linux"):
            backend, bname = cv2.CAP_V4L2, "V4L2"
        else:
            backend, bname = cv2.CAP_ANY, "Auto"

        log.info(f"Opening webcam {WEBCAM_INDEX} via {bname} …")
        cap = cv2.VideoCapture(WEBCAM_INDEX, backend)

        if not cap.isOpened():
            log.error(f"Failed to open webcam index {WEBCAM_INDEX}")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # discard stale frames

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log.info(f"Webcam open ✅  {w}×{h} @ {fps:.0f} fps")
        return cap

    def _loop(self):
        """Inference loop — runs on the background thread."""
        self._set_status("running")
        start_time    = time.time()
        frame_idx     = 0
        fps           = 0.0
        fps_timer     = time.time()
        fps_count     = 0
        person_count  = 0

        # Blank placeholder frame while stream warms up
        blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    log.warning("Frame read failed — retrying …")
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                fps_count += 1

                # Rolling FPS
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps       = fps_count / elapsed
                    fps_timer = time.time()
                    fps_count = 0

                # ── Frame skipping ──────────────────────────────────────
                # On skipped frames, publish last cached JPEG without
                # re-running inference — halves CPU usage.
                if frame_idx % SKIP_N != 0:
                    with self._lock:
                        self._stats["fps"]          = round(fps, 1)
                        self._stats["total_frames"] = frame_idx
                        self._stats["uptime_sec"]   = round(time.time() - start_time, 1)
                    continue

                # ── Resize ──────────────────────────────────────────────
                resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # ── YOLO inference ──────────────────────────────────────
                results = self._model(
                    resized,
                    classes=[PERSON_CLASS_ID],
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    half=False,
                    verbose=False,
                    device="cpu",
                )

                # ── Parse ───────────────────────────────────────────────
                boxes, confs = [], []
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        boxes.append((x1, y1, x2, y2))
                        confs.append(float(box.conf[0]))

                person_count = len(boxes)

                if person_count >= LOG_MIN_PERSONS:
                    log.info(
                        f"Frame {frame_idx:06d} | FPS {fps:.1f} | "
                        f"Persons: {person_count}"
                    )

                # ── Draw ─────────────────────────────────────────────────
                display = resized.copy()
                _draw_boxes(display, boxes, confs)
                _draw_hud(display, fps, person_count)

                # ── Encode to JPEG ───────────────────────────────────────
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                ok, buffer = cv2.imencode(".jpg", display, encode_params)
                if not ok:
                    continue
                jpeg_bytes = buffer.tobytes()

                # ── Publish (thread-safe) ────────────────────────────────
                with self._lock:
                    self._frame_bytes          = jpeg_bytes
                    self._stats["fps"]          = round(fps, 1)
                    self._stats["person_count"] = person_count
                    self._stats["total_frames"] = frame_idx
                    self._stats["uptime_sec"]   = round(time.time() - start_time, 1)

        except Exception as exc:
            log.exception(f"Detector thread crashed: {exc}")
            self._set_status("error", error=str(exc))
        finally:
            self._running = False
            self._set_status("stopped")
            log.info("Detector thread exited.")