# src/

This folder contains the main source code for the CV Threat Detection system.

- `config.py`: Central configuration for model paths, thresholds, device, and other settings.
- `inference.py`: Contains the `HornetDetector` class for loading the YOLO model and running predictions.
- `main_testing.py`: Script for testing single-feed inference without the Streamlit UI.
- `multifeed.py`: Utilities for splitting and combining video feeds (e.g., for 4-feed display).
- `postprocessing.py`: Functions for drawing bounding boxes and formatting detection results.
- `streamlit_app.py`: Main Streamlit dashboard for running the multi-feed detection UI.
- `__pycache__/`: Python bytecode cache (should be ignored in version control).

**Relation to project:**
Implements the detection pipeline, UI, and all core logic. This is the heart of the project.