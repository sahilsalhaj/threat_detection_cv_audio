#!/usr/bin/env python3
"""
inferencing_cnn_test_only.py

Runs inference ONLY on the TEST split using the SAME
parent split logic + random seed as CNN training.

Guarantees:
- No re-splitting drift
- No data leakage
- Correct parent grouping
- Correct true labels
- Segment → parent mean aggregation
- Correct CNN input shapes
"""

import os
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# HARD-CODED PATHS
# =========================
DATA_DIR = "training_data"
CSV_PATH = "training_data/all_data_updated.csv"
RESULTS_DIR = "results/cnn_results"
MODEL_PATH = "models/cnn_models/cnn_best_20251213_005707.keras"

SEED = 1337
TEST_SIZE = 0.15

SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_LEN = 128

THRESHOLD = 0.5


# =========================
# Utilities
# =========================
def log(msg):
    print(f"[CNN-INF] {msg}")


# =========================
# Audio → log-mel (IDENTICAL to training)
# =========================
def make_log_mel(wave):
    mel = librosa.feature.melspectrogram(
        y=wave,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)

    if log_mel.shape[1] < TARGET_LEN:
        pad = TARGET_LEN - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    else:
        log_mel = log_mel[:, :TARGET_LEN]

    return log_mel.astype(np.float32)


# =========================
# EXACT SAME TEST SPLIT AS TRAINING
# =========================
def build_test_split(csv_df, segment_files):
    df = csv_df.copy()
    df["health"] = (df["queen acceptance"] == 2).astype(int)

    parent_to_segments = defaultdict(list)
    for f in segment_files:
        if "__segment" in f:
            parent = f.split("__segment")[0] + ".raw"
            parent_to_segments[parent].append(f)

    parents, labels, file_lists = [], [], []

    for _, row in df.iterrows():
        p = row["file name"]
        if p in parent_to_segments:
            parents.append(p)
            labels.append(int(row["health"]))
            file_lists.append(sorted(parent_to_segments[p]))

    parents = np.array(parents)
    labels = np.array(labels)

    idx = np.arange(len(parents))
    _, idx_test = train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SEED
    )

    log(f"Using TEST parents: {len(idx_test)}")
    return parents, labels, file_lists, idx_test


# =========================
# Load spectrograms for TEST segments only
# =========================
def compute_specs(segment_files):
    spec_map = {}

    for i, fname in enumerate(segment_files):
        path = os.path.join(DATA_DIR, fname)
        try:
            audio, sr_file = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr_file != SR:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr_file,
                    target_sr=SR
                )

            spec = make_log_mel(audio)
            spec_map[fname] = spec

        except Exception as e:
            log(f"Failed {fname}: {e}")

        if (i + 1) % 200 == 0:
            log(f"Processed {i+1}/{len(segment_files)}")

    return spec_map


# =========================
# Build parent groups (CNN version)
# =========================
def build_parent_groups(spec_map, csv_df):
    parent_to_segments = defaultdict(list)

    for seg in spec_map:
        parent = seg.split("__segment")[0] + ".raw"
        parent_to_segments[parent].append(seg)

    csv_df = csv_df.copy()
    csv_df["health"] = (csv_df["queen acceptance"] == 2).astype(int)
    label_map = dict(zip(csv_df["file name"], csv_df["health"]))

    parents = sorted(parent_to_segments.keys())

    X_groups = []
    y_true = []
    counts = []

    for parent in parents:
        segs = sorted(parent_to_segments[parent])
        counts.append(len(segs))

        specs = np.stack(
            [spec_map[s] for s in segs],
            axis=0
        )[..., np.newaxis]   # (N, 128, 128, 1)

        X_groups.append(specs)
        y_true.append(label_map.get(parent, -1))

    return parents, X_groups, np.array(y_true), np.array(counts)


# =========================
# MAIN
# =========================
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("Loading CSV")
    csv_df = pd.read_csv(CSV_PATH)

    segment_files = sorted(
        f for f in os.listdir(DATA_DIR) if f.endswith(".wav")
    )

    parents, labels, file_lists, idx_test = build_test_split(
        csv_df, segment_files
    )

    # Collect TEST segments only
    test_segments = []
    for i in idx_test:
        test_segments.extend(file_lists[i])

    test_segments = sorted(set(test_segments))
    log(f"Total TEST segments: {len(test_segments)}")

    log("Computing spectrograms")
    spec_map = compute_specs(test_segments)

    log("Building parent groups")
    parents_t, X_test, y_test, counts = build_parent_groups(
        spec_map, csv_df
    )

    log("Loading CNN model")
    model = tf.keras.models.load_model(MODEL_PATH)

    log("Running inference (segment → parent mean)")
    scores, preds = [], []

    for specs in X_test:
        seg_scores = model.predict(specs, verbose=0).ravel()
        mean_score = float(np.mean(seg_scores))
        scores.append(mean_score)
        preds.append(int(mean_score >= THRESHOLD))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(
        RESULTS_DIR, f"cnn_test_inference_{ts}.csv"
    )

    pd.DataFrame({
        "parent": parents_t,
        "probability_healthy": scores,
        "prediction": preds,
        "true_label": y_test,
        "segment_count": counts
    }).to_csv(out_csv, index=False)

    log(f"Saved → {out_csv}")

    print("\n===== CNN TEST PERFORMANCE =====")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nInference complete.")


if __name__ == "__main__":
    main()
