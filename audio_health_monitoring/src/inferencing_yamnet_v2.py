#!/usr/bin/env python3
"""
inferencing_yamnet_v2.py

Inference script for the trained YAMNet v2 sequence model.

Features:
 - Automatically loads best checkpoint path + best threshold from metrics JSON
 - Loads/caches segment embeddings for the demo folder
 - Reconstructs parent sequences exactly like training
 - Runs inference on each parent
 - Saves results to results/demo_inference_<timestamp>.csv
 - Optional: provide CSV to evaluate accuracy on demo set

Usage:
  python src/inferencing_yamnet_v2.py --demo_dir demo_testing --csv training_data/all_data_updated.csv

Arguments:
  --demo_dir     Folder containing .wav segments for demo/real inference
  --csv          (Optional) CSV with ground truth to evaluate demo accuracy
  --model_path   Path to model (.keras). If not provided, auto-loads best checkpoint
  --threshold    Override threshold (otherwise read from metrics JSON)
"""

import os
import argparse
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------
# Utility
# ---------------------------------------------------
def log(msg):
    print(f"[INF] {msg}")


# ---------------------------------------------------
# Load segment embeddings (same as training)
# ---------------------------------------------------
def compute_embeddings_for_folder(folder, yamnet_handle, sr=16000):
    """
    Computes embeddings for ALL .wav files in the folder.
    Returns dict {filename: 1024-d embedding}
    """
    yamnet = hub.load(yamnet_handle)
    log("Loaded YAMNet for inference.")

    seg_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
    log(f"Found {len(seg_files)} segment files.")

    mapping = {}
    for i, seg in enumerate(seg_files):
        full = os.path.join(folder, seg)
        try:
            audio, file_sr = sf.read(full, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if file_sr != sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            audio = audio.astype(np.float32)

            scores, embeddings, _ = yamnet(audio)
            emb_avg = tf.reduce_mean(embeddings, axis=0).numpy()
            mapping[seg] = emb_avg
        except Exception as e:
            log(f"Failed reading {seg}: {e}")

        if (i + 1) % 100 == 0:
            log(f"Processed {i+1}/{len(seg_files)}")

    return mapping


# ---------------------------------------------------
# Build parent sequences (same format as training)
# ---------------------------------------------------
def build_parent_sequences(mapping, csv_df=None, max_segments_cap=12):
    """
    Groups segments into parents using filename patterns like:
        2022-06-09--12-51-03_1__segment4.wav
    Parent key = <prefix>.raw

    Returns:
        parents, X_seq, lengths, true_labels(optional), file_lists
    """
    parent_to_segments = defaultdict(list)

    for seg in mapping.keys():
        if "__segment" not in seg:
            continue
        parent = seg.split("__segment")[0] + ".raw"
        parent_to_segments[parent].append(seg)

    parents = sorted(parent_to_segments.keys())
    file_lists = []
    lengths = []

    # CSV label mapping if provided
    label_map = None
    if csv_df is not None and "file name" in csv_df.columns and "queen acceptance" in csv_df.columns:
        csv_df = csv_df.copy()
        csv_df["health"] = (csv_df["queen acceptance"] == 2).astype(int)
        label_map = dict(zip(csv_df["file name"], csv_df["health"]))

    # build sequences
    max_segments = min(
        max(len(v) for v in parent_to_segments.values()),
        max_segments_cap
    )
    log(f"max_segments used = {max_segments}")

    X = np.zeros((len(parents), max_segments, 1024), dtype=np.float32)
    y_true = []

    for i, parent in enumerate(parents):
        segs = sorted(parent_to_segments[parent])[:max_segments]
        file_lists.append(segs)
        lengths.append(len(segs))

        # fill sequence
        for j, seg in enumerate(segs):
            X[i, j] = mapping[seg]

        # label
        if label_map:
            y_true.append(label_map.get(parent, -1))
        else:
            y_true.append(-1)  # unknown

    return parents, X, np.array(lengths), np.array(y_true), file_lists


# ---------------------------------------------------
# Load model + threshold
# ---------------------------------------------------
def load_model_and_threshold(model_path, metrics_json_path, manual_threshold):
    threshold = None

    if manual_threshold is not None:
        threshold = manual_threshold
        log(f"Using manual threshold: {threshold}")

    # auto detect threshold from metrics JSON
    elif os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            m = json.load(f)
        threshold = m.get("best_threshold", 0.5)
        log(f"Loaded best_threshold={threshold} from metrics JSON")
    else:
        threshold = 0.5
        log("No metrics JSON found. Using default threshold=0.5")

    model = tf.keras.models.load_model(model_path)
    log(f"Loaded model: {model_path}")

    return model, threshold


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, default="demo_testing")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--metrics_json", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--yamnet_handle", type=str, default="https://tfhub.dev/google/yamnet/1")
    parser.add_argument("--max_segments_cap", type=int, default=12)
    args = parser.parse_args()

    # auto-detect metrics summary JSON if not given
    if args.metrics_json is None:
        metrics_files = sorted(
            [f for f in os.listdir("results") if f.startswith("metrics_summary_v2")],
            reverse=True
        )
        if len(metrics_files) == 0:
            raise RuntimeError("No metrics_summary_v2 JSON found in results/.")
        args.metrics_json = os.path.join("results", metrics_files[0])
        log(f"Auto-selected metrics: {args.metrics_json}")

    # auto-detect best model if not provided
    if args.model_path is None:
        with open(args.metrics_json, "r") as f:
            m = json.load(f)
        best_path = m.get("best_ckpt_path")
        if best_path is None or not os.path.exists(best_path):
            raise RuntimeError("Best checkpoint not found. Provide --model_path manually.")
        args.model_path = best_path
        log(f"Auto-selected model: {args.model_path}")

    # load CSV for evaluation
    csv_df = pd.read_csv(args.csv) if args.csv else None

    # Compute demo embeddings
    mapping = compute_embeddings_for_folder(args.demo_dir, args.yamnet_handle)

    # Build parent sequences (same as training)
    parents, X, lengths, y_true, file_lists = build_parent_sequences(
        mapping, csv_df, max_segments_cap=args.max_segments_cap
    )

    # Load model + threshold
    model, threshold = load_model_and_threshold(args.model_path, args.metrics_json, args.threshold)

    # Predict
    log("Running inference...")
    scores = model.predict(X).ravel()
    preds = (scores >= threshold).astype(int)

    # Output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"results/demo_inference_{timestamp}.csv"

    df = pd.DataFrame({
        "parent": parents,
        "probability_healthy": scores,
        "pred_label": preds,
        "true_label": y_true,
        "segment_count": lengths
    })
    df.to_csv(out_csv, index=False)

    log(f"Saved detailed inference results → {out_csv}")

    # If ground-truth exists, print accuracy metrics
    if csv_df is not None:
        valid_idx = y_true != -1
        if valid_idx.sum() > 0:
            print("\n===== DEMO SET EVALUATION =====")
            print(classification_report(y_true[valid_idx], preds[valid_idx]))
            print("Confusion Matrix:")
            print(confusion_matrix(y_true[valid_idx], preds[valid_idx]))

    print("\nInference complete!")


if __name__ == "__main__":
    main()
