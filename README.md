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

This project evaluates **beehive colony health using audio signals**, with the goal of classifying **healthy vs unhealthy (queen acceptance failure)** hives from continuous hive sounds.

The system implements **two complementary audio pipelines**:

- **YAMNet v2 (Embedding + Sequence Model)**
  - Uses **pretrained YAMNet** to extract 1024-dim embeddings from 1-second audio segments  
  - Groups segments by hive recording (“parent”)  
  - Applies a **BiLSTM-based sequence classifier** for parent-level prediction  
  - Includes threshold tuning, ROC/PR analysis, and full artifact logging  

- **CNN on Log-Mel Spectrograms (Direct Audio Learning)**
  - Trains a **custom 2D CNN** directly on log-mel spectrograms  
  - Performs **segment-level inference** with **parent-level aggregation**  
  - Avoids pretrained embeddings, enabling task-specific feature learning  
  - Achieved **significantly higher accuracy** than the YAMNet pipeline on this dataset
    


Refer to the `audio_health_monitoring` folder for:
- Training & inference scripts (`src/`)
- Audio datasets (`training_data/`)
- Trained models (`models/`)
- Evaluation metrics and artifacts (`results/`)
