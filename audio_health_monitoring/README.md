# 🐝 Apiculture Audio Health Monitoring

A production-ready system that **listens to beehive audio** and predicts hive health using deep learning.  
The objective is precise and measurable: classify a hive as **healthy** (queen accepted) or **unhealthy** based purely on its acoustic signature.

This module contains **two independent audio pipelines**, ordered from most task-specific to most general.

---

## 🎧 What the System Does

Beehives generate continuous low-frequency hums whose structure changes with colony condition.  
This project converts those raw sounds into **parent-level health predictions** by:

- Segmenting long hive recordings into short audio clips  
- Learning acoustic representations using deep neural networks  
- Aggregating segment-level predictions into robust hive-level decisions  
- Producing fully interpretable evaluation artifacts for analysis and reporting  

---

## 🧠 Models Implemented

### 1️⃣ CNN on Log-Mel Spectrograms (Primary Model)

A **custom convolutional neural network** trained end-to-end on beehive audio.

**Pipeline**
- Raw audio → log-mel spectrogram (128×128)
- Spectrogram → deep CNN
- Segment-level predictions → parent-level mean aggregation

**Key Properties**
- No pretrained audio model dependency  
- Learns task-specific frequency patterns directly from hive sounds  
- Segment-level inference with parent-level aggregation  
- Deterministic train/val/test splits (no leakage)  
- Achieved **~96–97% accuracy**, outperforming YAMNet on this dataset  

**Why it matters**
This model captures **subtle hive-specific acoustic cues** that generic audio classifiers often miss.

---

### 2️⃣ YAMNet v2 + Sequence Model (Reference Model)

A hybrid architecture combining pretrained embeddings with temporal modeling.

**Pipeline**
- Raw audio → **YAMNet embeddings (1024-D per second)**  
- Segments grouped by hive recording (“parent”)  
- Parent sequence → **BiLSTM + Dense classifier**  
- Validation-based threshold tuning  

**Key Properties**
- Leverages Google’s pretrained YAMNet for feature extraction  
- Explicit temporal modeling using BiLSTM  
- Strong baseline with full metric logging and visualizations  

**Why it exists**
Provides a **generalized, transferable baseline** and a comparison point for the CNN approach.

---

## 🚀 Usage Overview

1. **Prepare the dataset**
   - `.wav` audio segments  
   - CSV with hive metadata and queen acceptance labels  

2. **Train a model**
   - Run the CNN or YAMNet training script  
   - Automatic splitting, caching, evaluation, and artifact saving  

3. **Run inference**
   - Evaluate strictly on the saved test split  
   - Generate parent-level predictions and reports  

4. **Analyze results**
   - Accuracy, precision, recall, F1  
   - ROC & PR curves  
   - Confusion matrices  
   - Prediction CSVs  

---

## 🔍 How It Works (Short)

**CNN Path**
- Audio → spectrogram → convolutional feature extraction  
- Learns frequency-time patterns unique to hive health  
- Aggregates multiple segments into a single hive prediction  

**YAMNet Path**
- Audio → pretrained embeddings  
- Sequences capture long-term hive rhythm  
- BiLSTM models temporal structure  

Both pipelines enforce:
- Parent-level stratification  
- Fixed random seeds  
- No train/test contamination  

---

## 📈 What You Get

After training, the system automatically produces:

- Trained models (`.keras`)  
- Test-set predictions  
- Classification reports (CSV)  
- Metrics summaries (JSON)  
- ROC / PR curves  
- Confusion matrices  
- Training history plots  
- Cached embeddings or spectrograms for fast reruns  

## Folder Structure
```
apiculture_final/
└── audio_health_monitoring/
    ├── src/
    │   ├── train_cnn.py
    │   ├── inferencing_cnn.py
    │   ├── train_yamnet_v2.py
    │   ├── inferencing_yamnet_v2.py
    │
    ├── training_data/
    │   ├── all_data_updated.csv
    │   └── *.wav
    │
    ├── models/
    │   ├── cnn_models/
    │   └── yamnet_models/
    │
    └── results/
        ├── cnn_results/
        └── yamnet_results/

```

---

## 🌱 Why It Matters

Bee health directly affects pollination, food security, and ecosystems.  
This system enables **non-invasive, low-cost, early detection of hive issues** using nothing more than sound — scalable from research labs to real apiaries.
