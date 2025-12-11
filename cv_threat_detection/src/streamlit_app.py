import streamlit as st
import cv2
import numpy as np
import sys
import os

# ---------------------------------------------------------
# FIX IMPORT PATHS so that cv_threat_detection package loads
# ---------------------------------------------------------
FILE = os.path.abspath(__file__)                  # .../cv_threat_detection/src/streamlit_app.py
SRC_DIR = os.path.dirname(FILE)                   # .../cv_threat_detection/src
PKG_DIR = os.path.dirname(SRC_DIR)                # .../cv_threat_detection
PROJECT_ROOT = os.path.dirname(PKG_DIR)           # .../apiculture_final

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("DEBUG: Python exe =", sys.executable)
print("DEBUG: cwd =", os.getcwd())
print("DEBUG: PROJECT_ROOT added to sys.path =", PROJECT_ROOT)

# ---------------------------------------------------------
# IMPORT YOUR MODULES
# ---------------------------------------------------------
from cv_threat_detection.src.inference import HornetDetector
from cv_threat_detection.src.multifeed import split_into_four
from cv_threat_detection.src.postprocessing import draw_results
from cv_threat_detection.src.config import SOURCE

# Blank placeholder so captions don't error
BLANK = np.zeros((50, 50, 3), dtype=np.uint8)

# ---------------------------------------------------------
# STREAMLIT UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Apiculture Threat Monitor", layout="wide")

st.title("🐝 Apiculture Threat Detection Dashboard (4-Feed View)")
st.write("Live YOLO detection from webcam, arranged in a clean 2×2 view.")

# Create 2×2 layout
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

FRAME_WINDOW_1 = row1_col1.image(BLANK, caption="Feed A")
FRAME_WINDOW_2 = row1_col2.image(BLANK, caption="Feed B")
FRAME_WINDOW_3 = row2_col1.image(BLANK, caption="Feed C")
FRAME_WINDOW_4 = row2_col2.image(BLANK, caption="Feed D")

st.sidebar.header("Settings")
enable_detection = st.sidebar.toggle("Enable YOLO Detection", value=True)
round_robin = st.sidebar.toggle("Round Robin Inference (Recommended)", value=True)

st.sidebar.write("---")
st.sidebar.write("Press **Stop** in Streamlit to end the session.")

# ---------------------------------------------------------
# YOLO DETECTOR INIT
# ---------------------------------------------------------
detector = HornetDetector()

# ---------------------------------------------------------
# WEBCAM SOURCE
# ---------------------------------------------------------
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    st.error("❌ Could not open webcam/video source.")
    st.stop()

idx = 0  # which feed to process during round-robin


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Failed to receive frame from camera.")
        break

    # 1: Split into quadrants
    feeds = split_into_four(frame)
    # feeds = [TL, TR, BL, BR]

    # 2: YOLO processing
    if enable_detection:
        if round_robin:
            results = detector.predict(feeds[idx])
            feeds[idx] = draw_results(feeds[idx], results)
            idx = (idx + 1) % 4
        else:
            for i in range(4):
                results = detector.predict(feeds[i])
                feeds[i] = draw_results(feeds[i], results)

    # 3: Convert for display
    feeds = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in feeds]

    # 4: Reorder feeds permanently (swap diagonals)
    display_feeds = {
        "TL": feeds[3],  # original BR
        "TR": feeds[2],  # original BL
        "BL": feeds[1],  # original TR
        "BR": feeds[0],  # original TL
    }

    # 5: Display in 2×2 grid
    FRAME_WINDOW_1.image(display_feeds["TL"], width=400, caption="Top-Left")
    FRAME_WINDOW_2.image(display_feeds["TR"], width=400, caption="Top-Right")
    FRAME_WINDOW_3.image(display_feeds["BL"], width=400, caption="Bottom-Left")
    FRAME_WINDOW_4.image(display_feeds["BR"], width=400, caption="Bottom-Right")

