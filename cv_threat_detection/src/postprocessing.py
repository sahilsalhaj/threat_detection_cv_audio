# import cv2
# from .config import DRAW_BOXES, DRAW_LABELS

# # -----------------------------
# # NEW FILTER SETTINGS
# # -----------------------------
# MAX_BOX_RATIO = 0.85   # discard boxes covering >85% of width/height
# MIN_BOX_RATIO = 0.01   # optional: discard boxes smaller than 1% of width/height


# def draw_results(frame, results):
#     """ Draw bounding boxes + labels with edge and size filtering. """
#     if results is None or results.boxes is None:
#         return frame

#     h, w = frame.shape[:2]

#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         # -----------------------------
#         # Clip coordinates to frame
#         # -----------------------------
#         x1_clipped = max(0, min(w-1, x1))
#         y1_clipped = max(0, min(h-1, y1))
#         x2_clipped = max(0, min(w-1, x2))
#         y2_clipped = max(0, min(h-1, y2))

#         box_w = x2_clipped - x1_clipped
#         box_h = y2_clipped - y1_clipped

#         # -----------------------------
#         # Filter boxes that are too large or too small
#         # -----------------------------
#         if box_w / w > MAX_BOX_RATIO or box_h / h > MAX_BOX_RATIO:
#             continue  # skip huge ghost boxes

#         if box_w / w < MIN_BOX_RATIO or box_h / h < MIN_BOX_RATIO:
#             continue  # skip tiny boxes (optional)

#         # -----------------------------
#         # Draw box
#         # -----------------------------
#         if DRAW_BOXES:
#             cv2.rectangle(frame, (x1_clipped, y1_clipped), (x2_clipped, y2_clipped), (0, 255, 0), 2)

#         # -----------------------------
#         # Draw label
#         # -----------------------------
#         if DRAW_LABELS:
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])
#             label = f"{results.names[cls]} {conf:.2f}"
#             cv2.putText(frame, label, (x1_clipped, y1_clipped-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                         (0, 255, 0), 2)

#     return frame



# postprocessing.py

import cv2
from .config import DRAW_BOXES, DRAW_LABELS

# Thresholds for filtering boxes
MAX_BOX_RATIO = 0.8   # discard boxes wider or taller than 80% of frame
EDGE_MARGIN = 5       # pixels considered "touching" the edge

def draw_results(frame, results):
    """ Draw bounding boxes + labels, with filtering for large boxes and corner-edge ghosts. """
    if results is None or results.boxes is None:
        return frame

    h_frame, w_frame = frame.shape[:2]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        # 1️⃣ Filter: discard boxes that are too large
        if width / w_frame > MAX_BOX_RATIO or height / h_frame > MAX_BOX_RATIO:
            continue

        # 2️⃣ Filter: discard boxes that touch two adjacent edges (corner ghost)
        touching_left = x1 <= EDGE_MARGIN
        touching_right = x2 >= w_frame - EDGE_MARGIN
        touching_top = y1 <= EDGE_MARGIN
        touching_bottom = y2 >= h_frame - EDGE_MARGIN

        if (touching_left and touching_top) or \
           (touching_left and touching_bottom) or \
           (touching_right and touching_top) or \
           (touching_right and touching_bottom):
            continue

        # Draw the box
        if DRAW_BOXES:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the label
        if DRAW_LABELS:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{results.names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    return frame
