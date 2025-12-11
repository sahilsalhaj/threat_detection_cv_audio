# inference.py

from ultralytics import YOLO
import torch
import numpy as np
from .config import (
    MODEL_PATH, DEVICE, CONF_THRESHOLD, IOU_THRESHOLD,
    IMG_SIZE
)


class HornetDetector:
    def __init__(self):
        # Decide device: try CUDA first, fallback to CPU
        if DEVICE.lower() in ["cuda", "gpu", "0"]:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("[INFO] Using GPU (CUDA)")
            else:
                self.device = "cpu"
                print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        else:
            # User explicitly chose CPU in config
            self.device = "cpu"
            print("[INFO] Using CPU")

        print(f"[INFO] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

    def predict(self, frame):
        """ Run prediction on a single frame. """

        results = self.model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=self.device
        )
        return results[0]

