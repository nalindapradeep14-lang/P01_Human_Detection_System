# =============================================================================
# VisionTrack Edu — Flask Web Server
# =============================================================================
#
# QUICK START
# -----------
#   pip install -r requirements.txt
#   python app.py
#   Open browser → http://localhost:5000
#
# ENDPOINTS
# ---------
#   GET  /                → Web dashboard (HTML)
#   GET  /video_feed      → MJPEG live stream
#   POST /start           → Start detection
#   POST /stop            → Stop detection
#   GET  /stats           → JSON stats (fps, count, uptime, status)
#
# =============================================================================

import logging
import logging.handlers
import time
from flask import Flask, Response, jsonify, render_template

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, ENABLE_FILE_LOGGING, LOG_FILE_PATH
from detector import HumanDetector

# =============================================================================
# LOGGING
# =============================================================================

def setup_logger():
    fmt    = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger("VisionTrack")
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if ENABLE_FILE_LOGGING:
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=2_000_000, backupCount=3
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

setup_logger()
log = logging.getLogger("VisionTrack.App")

# =============================================================================
# FLASK APP
# =============================================================================

app      = Flask(__name__)
detector = HumanDetector()

# =============================================================================
# MJPEG STREAM GENERATOR
# =============================================================================

# A tiny static JPEG placeholder shown while the detector is stopped.
# Generated once at startup so we never send an empty response.
import cv2, numpy as np

def _make_placeholder(text: str = "Detection stopped") -> bytes:
    """Return a JPEG bytes object showing a dark 'offline' placeholder."""
    img = np.zeros((416, 416, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)                       # dark background

    # Grid lines for a surveillance-monitor feel
    for x in range(0, 416, 52):
        cv2.line(img, (x, 0), (x, 416), (30, 30, 38), 1)
    for y in range(0, 416, 52):
        cv2.line(img, (0, y), (416, y), (30, 30, 38), 1)

    # Icon-like circle
    cv2.circle(img, (208, 170), 40, (0, 60, 30), 2)
    cv2.line(img, (208, 130), (208, 210), (0, 180, 80), 2)
    cv2.line(img, (168, 170), (248, 170), (0, 180, 80), 2)

    # Text
    cv2.putText(img, text, (208 - len(text) * 6, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "Press START to begin", (95, 278),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 70), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


PLACEHOLDER_BYTES = _make_placeholder()


def generate_mjpeg():
    """
    Generator that yields a continuous MJPEG stream.

    Each chunk is a complete JPEG frame wrapped in the multipart boundary
    that browsers understand as Motion JPEG (MJPEG).

    When the detector is stopped, the placeholder frame is yielded at ~5 fps
    so the <img> tag never goes blank.
    """
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"

    while True:
        frame = detector.get_frame()

        if frame is None:
            # Detector not running — send placeholder
            data = PLACEHOLDER_BYTES
            time.sleep(0.2)
        else:
            data = frame

        yield boundary + data + b"\r\n"


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    MJPEG stream endpoint.
    The browser's <img src="/video_feed"> connects here and receives a
    continuous stream of JPEG frames wrapped in multipart boundaries.
    """
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/start", methods=["POST"])
def start():
    """Start the detector. Returns JSON status."""
    if detector.is_running():
        return jsonify({"ok": False, "message": "Detector already running."})

    success = detector.start()
    if success:
        log.info("Detection started via web UI.")
        return jsonify({"ok": True, "message": "Detection started."})
    else:
        stats = detector.get_stats()
        return jsonify({"ok": False, "message": stats.get("error", "Start failed.")}), 500


@app.route("/stop", methods=["POST"])
def stop():
    """Stop the detector. Returns JSON status."""
    if not detector.is_running():
        return jsonify({"ok": False, "message": "Detector not running."})

    detector.stop()
    log.info("Detection stopped via web UI.")
    return jsonify({"ok": True, "message": "Detection stopped."})


@app.route("/stats")
def stats():
    """
    Return live stats as JSON — polled by the dashboard every second.

    Response shape:
        {
          "fps":          14.3,
          "person_count": 2,
          "total_frames": 1042,
          "uptime_sec":   73.4,
          "status":       "running"   // stopped | starting | running | error
        }
    """
    return jsonify(detector.get_stats())


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    log.info(f"VisionTrack Edu Web — http://{FLASK_HOST}:{FLASK_PORT}")
    log.info("Open your browser at  http://localhost:5000")
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,     # Required: MJPEG stream + API calls must run concurrently
        use_reloader=False # Prevent double-loading the YOLO model on debug reload
    )