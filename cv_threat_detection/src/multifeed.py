# multifeed.py

import cv2
import numpy as np

def split_into_four(frame):
    """Split frame into 4 equal quadrants: TL, TR, BL, BR."""
    h, w = frame.shape[:2]
    half_h = h // 2
    half_w = w // 2

    top_left     = frame[0:half_h, 0:half_w]
    top_right    = frame[0:half_h, half_w:w]
    bottom_left  = frame[half_h:h, 0:half_w]
    bottom_right = frame[half_h:h, half_w:w]

    return [top_left, top_right, bottom_left, bottom_right]


def combine_four(q1, q2, q3, q4, pad=10):
    """Combine feeds horizontally with spacing."""
    # Create vertical padding for uniform height
    h1 = max(q1.shape[0], q2.shape[0], q3.shape[0], q4.shape[0])
    
    # Create blank spacer
    spacer = 255 * np.ones((h1, pad, 3), dtype=np.uint8)

    # Resize (optional: enforce equal height)
    feeds = [q1, q2, q3, q4]
    feeds = [cv2.resize(f, (f.shape[1], h1)) for f in feeds]

    # Combine: f1 | space | f2 | space | f3 | space | f4
    combined = cv2.hconcat([
        feeds[0], spacer,
        feeds[1], spacer,
        feeds[2], spacer,
        feeds[3]
    ])

    return combined
