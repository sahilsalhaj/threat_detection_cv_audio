# THIS FILE IS ONLY FOR TESTING SINGLE-FEED INFERENCE WITHOUT STREAMLIT, NOT PART OF FINAL PROJECT

import sys
import os
import cv2

# ---------------------------------------------------------
# FIX IMPORT PATHS (same logic as in streamlit_app.py)
# ---------------------------------------------------------
FILE = os.path.abspath(__file__)                  # .../cv_threat_detection/src/main.py
SRC_DIR = os.path.dirname(FILE)                   # .../cv_threat_detection/src
PKG_DIR = os.path.dirname(SRC_DIR)                # .../cv_threat_detection
PROJECT_ROOT = os.path.dirname(PKG_DIR)           # .../apiculture_final

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Debug prints (optional)
print("DEBUG main.py: Using Python =", sys.executable)
print("DEBUG main.py: PROJECT_ROOT =", PROJECT_ROOT)

# ---------------------------------------------------------
# IMPORT AFTER FIXING PATH
# ---------------------------------------------------------
from cv_threat_detection.src.inference import HornetDetector
from cv_threat_detection.src.postprocessing import draw_results
from cv_threat_detection.src.config import SOURCE, SHOW_WINDOW, SAVE_OUTPUT, OUTPUT_PATH


def main():
    print("[INFO] Initializing HornetDetector...")
    detector = HornetDetector()

    print(f"[INFO] Opening video source: {SOURCE}")
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam or video file.")
        return

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0

        print(f"[INFO] Saving output to {OUTPUT_PATH} ({width}x{height}, {fps} FPS)")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("[INFO] Running single-feed YOLO inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or camera disconnected.")
            break

        results = detector.predict(frame)
        frame_out = draw_results(frame, results)

        if SHOW_WINDOW:
            cv2.imshow("Threat Detection (Single Feed)", frame_out)

        if SAVE_OUTPUT and writer:
            writer.write(frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference finished.")


if __name__ == "__main__":
    main()
