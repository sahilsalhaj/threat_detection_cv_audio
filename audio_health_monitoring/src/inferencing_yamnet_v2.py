#!/usr/bin/env python3
"""
inferencing_yamnet_v2_test_only.py

Runs inference ONLY on the TEST split using the SAME
parent split logic + random seed as training.

No CLI args.
No re-splitting drift.
No data leakage.
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
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# HARD-CODED PATHS
# =========================
DATA_DIR = "training_data"
CSV_PATH = "training_data/all_data_updated.csv"
RESULTS_DIR = "results"
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

SEED = 1337
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MAX_SEGMENTS_CAP = 12


# =========================
# Utilities
# =========================
def log(msg):
    print(f"[INF] {msg}")


# =========================
# Load best model + threshold
# =========================
def load_best_model_and_threshold():
    metrics_files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.startswith("metrics_summary_v2")],
        reverse=True
    )
    if not metrics_files:
        raise RuntimeError("No metrics_summary_v2 JSON found")

    metrics_path = os.path.join(RESULTS_DIR, metrics_files[0])
    with open(metrics_path, "r") as f:
        m = json.load(f)

    model_path = m["best_ckpt_path"]
    threshold = m.get("best_threshold", 0.5)

    model = tf.keras.models.load_model(model_path)

    log(f"Loaded model: {model_path}")
    log(f"Using threshold: {threshold}")

    return model, threshold


# =========================
# Compute YAMNet embeddings
# =========================
def compute_embeddings(files, sr=16000):
    yamnet = hub.load(YAMNET_HANDLE)
    log("Loaded YAMNet")

    mapping = {}

    for i, fname in enumerate(files):
        full = os.path.join(DATA_DIR, fname)
        audio, file_sr = sf.read(full, dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        _, emb, _ = yamnet(audio)
        mapping[fname] = tf.reduce_mean(emb, axis=0).numpy()

        if (i + 1) % 100 == 0:
            log(f"Processed {i+1}/{len(files)}")

    return mapping


# =========================
# EXACT SAME SPLIT AS TRAINING
# =========================
def build_test_split(csv_df, segment_files):
    df = csv_df.copy()
    df["health"] = (df["queen acceptance"] == 2).astype(int)

    # Parent → segments
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

    idx_tmp, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )

    log(f"Using TEST parents: {len(idx_test)}")

    return parents, labels, file_lists, idx_test


# =========================
# Build parent sequences
# =========================
def build_parent_sequences(emb_map, csv_df, max_segments_cap=12):
    parent_to_segments = defaultdict(list)

    for seg in emb_map.keys():
        if "__segment" not in seg:
            continue
        parent = seg.split("__segment")[0] + ".raw"
        parent_to_segments[parent].append(seg)

    parents = sorted(parent_to_segments.keys())

    csv_df = csv_df.copy()
    csv_df["health"] = (csv_df["queen acceptance"] == 2).astype(int)
    label_map = dict(zip(csv_df["file name"], csv_df["health"]))

    X = []          # LIST (not numpy array)
    y_true = []
    counts = []

    for parent in parents:
        segs = sorted(parent_to_segments[parent])[:max_segments_cap]
        counts.append(len(segs))

        emb_seq = np.stack([emb_map[s] for s in segs], axis=0)
        X.append(emb_seq.astype(np.float32))

        y_true.append(label_map.get(parent, -1))

    return parents, X, np.array(y_true), np.array(counts)



# =========================
# MAIN
# =========================
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    csv_df = pd.read_csv(CSV_PATH)

    segment_files = sorted([
        f for f in os.listdir(DATA_DIR) if f.lower().endswith(".wav")
    ])

    parents, labels, file_lists, idx_test = build_test_split(
        csv_df, segment_files
    )

    # Collect only TEST segment files
    test_segment_files = []
    for i in idx_test:
        test_segment_files.extend(file_lists[i])

    test_segment_files = sorted(set(test_segment_files))

    log(f"Total TEST segments: {len(test_segment_files)}")

    emb_map = compute_embeddings(test_segment_files)

    parents_t, X_test, y_test, counts = build_parent_sequences(
        emb_map,
        csv_df,
        max_segments_cap=MAX_SEGMENTS_CAP
    )


    model, threshold = load_best_model_and_threshold()

    log("Running inference on TEST split...")
    scores = []
    preds = []

    for seq in X_test:
        seg_scores = model.predict(seq[np.newaxis, ...], verbose=0).ravel()
        mean_score = float(np.mean(seg_scores))
        scores.append(mean_score)
        preds.append(int(mean_score >= threshold))


    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(RESULTS_DIR, f"test_inference_{ts}.csv")

    pd.DataFrame({
        "parent": parents_t,
        "probability_healthy": scores,
        "prediction": preds,
        "true_label": y_test,
        "segment_count": counts
    }).to_csv(out_csv, index=False)

    log(f"Saved → {out_csv}")

    print("\n===== TEST SET PERFORMANCE =====")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
