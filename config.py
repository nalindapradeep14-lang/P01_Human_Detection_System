# =============================================================================
# VisionTrack Edu — Web Edition · Central Configuration
# =============================================================================

# --- Webcam ---
WEBCAM_INDEX   = 0
CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS    = 30

# --- Inference frame size (smaller = faster on CPU) ---
FRAME_WIDTH  = 416
FRAME_HEIGHT = 416

# --- Frame skipping (process every Nth frame) ---
SKIP_N = 2

# --- YOLOv8 model ---
MODEL_PATH           = "yolov8n.pt"
PERSON_CLASS_ID      = 0
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD        = 0.45

# --- Flask server ---
FLASK_HOST  = "0.0.0.0"   # 0.0.0.0 = accessible on your local network
FLASK_PORT  = 5000
FLASK_DEBUG = False

# --- MJPEG stream quality (1-100) ---
JPEG_QUALITY = 80

# --- Logging ---
ENABLE_FILE_LOGGING = True
LOG_FILE_PATH       = "visiontrack_web.log"
LOG_MIN_PERSONS     = 1

# --- Colours (BGR for OpenCV) ---
BOX_COLOR   = (0, 230, 100)
LABEL_COLOR = (0, 0, 0)