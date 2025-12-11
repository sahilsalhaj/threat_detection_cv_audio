
import os
# config.py

# -----------------------------
# MODEL + DEVICE
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8l.pt")   # change anytime
DEVICE = "cuda"                           # or "cpu"
IMG_SIZE = 640

# -----------------------------
# INFERENCE SETTINGS
# -----------------------------
CONF_THRESHOLD = 0.60
IOU_THRESHOLD = 0.45


# -----------------------------
# VIDEO / WEBCAM SETTINGS
# -----------------------------
SOURCE = 0                    # 0 = webcam, or "video.mp4"
SHOW_WINDOW = True            # Show GUI window
SAVE_OUTPUT = False
OUTPUT_PATH = "../results/output.mp4"

# -----------------------------
# DRAWING SETTINGS
# -----------------------------
DRAW_BOXES = True
DRAW_LABELS = True

# -----------------------------
# MISC
# -----------------------------
VERBOSE = True
