import streamlit as st
import cv2
import numpy as np
import sys
import os
import time

from send_notification import send_threat_notification


FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(FILE)
PKG_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(PKG_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("DEBUG: Python exe =", sys.executable)
print("DEBUG: cwd =", os.getcwd())
print("DEBUG: PROJECT_ROOT added to sys.path =", PROJECT_ROOT)


from cv_threat_detection.src.inference import HornetDetector
from cv_threat_detection.src.multifeed import split_into_four
from cv_threat_detection.src.postprocessing import draw_results
from cv_threat_detection.src.config import SOURCE


BLANK = np.zeros((50, 50, 3), dtype=np.uint8)


# ---------------- UI CONFIG ----------------

st.set_page_config(
    page_title="Apiculture Threat Monitor",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        h1 { font-size: 1.6rem; }
        h2 { font-size: 1.2rem; }
        .status-ok { color: #2e7d32; }
        .status-warn { color: #c62828; }
        .small { font-size: 0.85rem; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Apiculture Threat Monitoring")
st.markdown(
    "<span class='small'>Live multi-hive visual monitoring with automated threat alerts</span>",
    unsafe_allow_html=True,
)

st.write("")

# ---------------- LAYOUT ----------------

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

FRAME_WINDOW_1 = row1_col1.image(BLANK, caption="Hive 1")
FRAME_WINDOW_2 = row1_col2.image(BLANK, caption="Hive 2")
FRAME_WINDOW_3 = row2_col1.image(BLANK, caption="Hive 3")
FRAME_WINDOW_4 = row2_col2.image(BLANK, caption="Hive 4")


# ---------------- SIDEBAR ----------------

st.sidebar.header("System Controls")

enable_detection = st.sidebar.toggle("Enable detection", value=True)
round_robin = st.sidebar.toggle("Round-robin inference", value=True)

st.sidebar.write("---")
st.sidebar.subheader("Alert Status")

alert_box = st.sidebar.empty()


# ---------------- DETECTOR ----------------

detector = HornetDetector()


# ---------------- VIDEO SOURCE ----------------

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    st.error("Could not open video source.")
    st.stop()


# ---------------- HIVE CONFIG ----------------

HIVE_IDS = {
    0: "2b039978-0ae7-4063-8f38-b2c1e0532c21",
    1: "8a567d30-46b0-43be-813c-eb3883a213ed",
    2: "4b7338a7-cccf-45d0-843d-64ed15280ae8",
    3: "9c59996c-f941-441a-88ff-27b04fff6fd5",
}

DETECTION_HOLD_TIME = 1.5
ALERT_COOLDOWN = 60

threat_start_time = {hid: None for hid in HIVE_IDS.values()}
last_alert_time = {hid: 0 for hid in HIVE_IDS.values()}
armed = {hid: True for hid in HIVE_IDS.values()}

last_alert_ui = {
    "hive": None,
    "time": None,
    "status": None,
}

idx = 0


# ---------------- MAIN LOOP ----------------

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read frame from source.")
        break

    feeds = split_into_four(frame)

    if enable_detection:
        if round_robin:
            results = detector.predict(feeds[idx])

            hive_id = HIVE_IDS[idx]
            now = time.time()

            if results and len(results) > 0:
                if armed[hive_id]:
                    if threat_start_time[hive_id] is None:
                        threat_start_time[hive_id] = now

                    if (
                        now - threat_start_time[hive_id] >= DETECTION_HOLD_TIME
                        and now - last_alert_time[hive_id] >= ALERT_COOLDOWN
                    ):
                        annotated = draw_results(feeds[idx], results)
                        feeds[idx] = annotated

                        os.makedirs("alerts", exist_ok=True)
                        img_path = f"alerts/hive_{hive_id}.jpg"
                        cv2.imwrite(img_path, annotated)

                        ok = send_threat_notification(
                            image_path=img_path,
                            hive_id=hive_id,
                            threat_type="wasp",
                            confidence = float(results.boxes.conf[0])
                        )

                        last_alert_time[hive_id] = now
                        threat_start_time[hive_id] = None
                        armed[hive_id] = False

                        last_alert_ui["hive"] = hive_id
                        last_alert_ui["time"] = time.strftime("%H:%M:%S")
                        last_alert_ui["status"] = "sent" if ok else "failed"
                    else:
                        feeds[idx] = draw_results(feeds[idx], results)
                else:
                    feeds[idx] = draw_results(feeds[idx], results)
            else:
                threat_start_time[hive_id] = None
                armed[hive_id] = True
                feeds[idx] = draw_results(feeds[idx], results)

            idx = (idx + 1) % 4

        else:
            for i in range(4):
                results = detector.predict(feeds[i])
                feeds[i] = draw_results(feeds[i], results)

    feeds = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in feeds]

    display_feeds = {
        "TL": feeds[3],
        "TR": feeds[2],
        "BL": feeds[1],
        "BR": feeds[0],
    }

    FRAME_WINDOW_1.image(display_feeds["TL"], caption="Hive 1")
    FRAME_WINDOW_2.image(display_feeds["TR"], caption="Hive 2")
    FRAME_WINDOW_3.image(display_feeds["BL"], caption="Hive 3")
    FRAME_WINDOW_4.image(display_feeds["BR"], caption="Hive 4")

    if last_alert_ui["hive"]:
        alert_box.markdown(
            f"""
**Last Alert**
- Hive: `{last_alert_ui['hive']}`
- Time: `{last_alert_ui['time']}`
- Status: **{last_alert_ui['status']}**
"""
        )
    else:
        alert_box.markdown("_No alerts triggered yet_")
