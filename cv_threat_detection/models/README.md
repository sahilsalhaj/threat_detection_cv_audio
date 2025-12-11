# models/

This folder contains all model-related files for the CV Threat Detection project.

- `yolov11l.pt`, `yolov8l.pt`: Pre-trained YOLO model weights for hornet/wasp/bee detection. These are used for inference and can be swapped as needed in `config.py`.
- `get_model_info.py`: Utility script to print out model architecture, configuration, and training arguments for a given YOLO model file. Useful for debugging or understanding the model's structure.

**Relation to project:**
This folder is essential for storing the neural network weights and model utilities. The detection pipeline loads these weights to perform object detection on video feeds or images.