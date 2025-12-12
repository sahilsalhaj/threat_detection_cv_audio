
# Apiculture Audio Health Monitoring (YAMNet v2)

This project implements a production-ready audio-based beehive health classifier using **YAMNet embeddings + a sequence classifier**.  
It supports full training, evaluation, artifact logging, demos, and inference.

---

## 📦 Folder Structure

```
apiculture_final/
└── audio_health_monitoring/
    ├── src/
    │   ├── train_yamnet_v2.py
    │   ├── inferencing_yamnet_v2.py
    │   └── (other scripts)
    ├── training_data/
    │   ├── all_data_updated.csv
    │   ├── *.wav (segments)
    ├── demo_testing/
    ├── results/
    └── models/
```

---

## 🧰 Environment Setup

### **1. Create a virtual environment**

```
python -m venv audio_env
```

### **2. Activate it**

**Windows:**
```
audio_env\Scripts\activate
```

**Linux/macOS:**
```
source audio_env/bin/activate
```

---

## 🔧 Install Dependencies

Run this inside the virtual environment:

```
pip install tensorflow tensorflow-hub librosa soundfile scikit-learn matplotlib seaborn pandas numpy
```

---

## 🧠 How the System Works (Short Version)

1. **YAMNet** extracts 1024‑dim embeddings from each 1‑second frame of audio.  
2. We **average** embeddings per segment → each `.wav` segment becomes one vector.  
3. We **group segments into parents** (`<rawname>.raw`) to represent the full 60s recording.  
4. Each parent becomes a **sequence of embeddings**, up to 12 segments.  
5. A **sequence model** (Dense + BatchNorm + Dropout) predicts:
   - `1` → **Healthy hive**
   - `0` → **Unhealthy / queen not accepted**

6. During inference, the system reconstructs the same sequences and performs prediction with a **tuned threshold**.

---

## 🎯 Training (`train_yamnet_v2.py`)

This script:

- Builds a clean train/val/test split  
- Computes/caches YAMNet embeddings  
- Trains the sequence classifier  
- Performs early stopping  
- Logs:
  - ROC curve
  - PR curve
  - Confusion matrix
  - Classification report
  - Training curves
  - Best threshold
  - Best checkpoint
  - Final model
- Saves everything to `results/` and `models/`

### **Run Training:**

```
python src/train_yamnet_v2.py --data_dir training_data --csv all_data_updated.csv
```

Artifacts appear in:

```
results/
models/
```

---

## 🧪 Demo Inference (`inferencing_yamnet_v2.py`)

This script:

- Loads the **best checkpoint** automatically  
- Loads the **best threshold** from the metrics JSON  
- Computes embeddings for all `.wav` in `demo_testing/`
- Builds parent sequences exactly like training
- Predicts & saves results to:

```
results/demo_inference_<timestamp>.csv
```

### **Run Inference:**

```
python src/inferencing_yamnet_v2.py --demo_dir demo_testing --csv training_data/all_data_updated.csv
```

The CSV will include:

| parent | probability | prediction | true_label | segment_count |
|-------|-------------|------------|------------|----------------|

Perfect for project demonstrations.

---

## 📊 Understanding Your Results

### **1. AUC (ROC Curve)**
Measures how well the model separates healthy/unhealthy Hives.
- **0.90+ is strong**
- **0.95+ is excellent**

### **2. PR Curve**
Shows precision vs recall for the healthy class.  
AP above **0.90** means strong prediction reliability.

### **3. Confusion Matrix**
Shows actual vs predicted:
- Top-left: true unhealthy predicted unhealthy
- Bottom-right: true healthy predicted healthy

### **4. Classification Report**
Includes:
- Precision
- Recall
- F1‑Score
- Support (samples per class)

---

## 📁 Demo Testing Folder

You can place any new hive audio segments here:

```
demo_testing/
    2022-07-11--15-32-01_1__segment0.wav
    2022-07-11--15-32-01_1__segment1.wav
    ...
```

The inference script automatically:
- Groups them by parent
- Produces predictions

---

## 🧩 Troubleshooting

### **Not enough segments detected**
Ensure filenames follow:
```
<rawprefix>__segment<number>.wav
```

### **Model/threshold not found**
Ensure `results/metrics_summary_v2_*.json` exists (generated during training).

### **Accuracy lower than expected**
Possible reasons:
- Very noisy segments  
- Parent has too few segments  
- Large imbalance (handled via class weighting but may still vary)

---

## 🏁 Final Notes

You now have a **complete production pipeline**:
- Training  
- Evaluation  
- Artifact generation  
- Threshold tuning  
- Demo & inference  

Perfect for academic submission and real-world deployment.

