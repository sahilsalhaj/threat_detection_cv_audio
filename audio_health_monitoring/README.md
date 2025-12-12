# 🐝 Apiculture Audio Health Monitoring

A compact system that listens to beehive audio and predicts hive health using deep learning.  
The goal is simple: identify whether a hive is **healthy** (queen accepted) or **unhealthy** based purely on its sound profile.

---

## 🎧 What the System Does
Beehives produce distinct acoustic patterns depending on their internal state.  
This project turns those patterns into meaningful predictions by:

- Extracting audio features using **Google’s YAMNet**  
- Grouping segments from the same hive recording into a sequence  
- Feeding the sequence into a **BiLSTM model** trained to classify health  
- Automatically tuning thresholds for the best accuracy  
- Producing interpretable results: ROC curves, confusion matrix, metrics, and reports

---

## 🚀 Usage Overview
1. Prepare your dataset: a folder of waveform segments (.wav) and a CSV with hive metadata.  
2. Install requirements and run the training script to build the model.  
3. Use the inference script on new audio to generate health predictions.  
4. Optionally place a few example files in a demo folder to showcase predictions live.

---

## 🔍 How It Works (Short)
- YAMNet converts raw audio into a 1024-dimensional embedding.  
- Segments belonging to the same minute of audio are stitched together into a time series.  
- A recurrent model analyzes the “buzz rhythm” of the hive.  
- The system outputs a probability of the hive being healthy.

---

## 📈 What You Get
After training, the project automatically produces:

- Health classifier (.keras model)  
- ROC/PR curves, confusion matrix, accuracy/loss curves  
- Detailed metrics JSON  
- Prediction CSVs for test and demo data  
- A fully reproducible embedding cache for fast retraining  

---

## 🌱 Why It Matters
Healthy bees mean healthy crops.  
This system gives beekeepers and researchers a low-cost, real-time way to detect hive problems early — no invasive inspection required.

