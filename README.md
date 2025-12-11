# Apiculture Final Project

This repository contains two main sub-projects focused on modern beehive monitoring using computer vision and audio analysis:

## 1. CV Threat Detection

This project uses computer vision techniques to detect threats and monitor bee activity in and around the hive. Key features include:

- **YOLO-based Models:** Utilizes YOLOv8 and YOLOv11 models for object detection and threat identification.
- **Training & Evaluation:** Includes scripts and notebooks for training models, evaluating results, and visualizing metrics.
- **Streamlit App:** Provides an interactive interface for real-time inference and visualization.
- **Data Structure:** Organized training, validation, and test datasets for model development.
- **Results:** Evaluation metrics and sample outputs are stored for reference.

Refer to the `cv_threat_detection` folder for:
- Source code (`src/`)
- Model files (`models/`)
- Training data (`training_data/`)
- Evaluation results (`results/`)
- Notebooks for experimentation (`notebooks/`)

## 2. Audio Health Monitoring

This project aims to assess the health of bee colonies using audio signals recorded from beehives. The goal is to:

- **Train a Convolutional Network (CNN):** Use beehive audio recordings to teach a neural network to classify the health status of the colony (healthy or unhealthy).
- **Future Work:** Data collection, preprocessing, model training, and evaluation will be implemented here.

Refer to the `audio_health_monitoring` folder for:
- Planned code and models (`src/`, `models/`)
- Training data (`training_data/`)
- Results and notebooks (`results/`, `notebooks/`)

---

This project is designed to provide a comprehensive toolkit for apiculture monitoring using both visual and audio data. For more details, see the README files in each sub-project folder.