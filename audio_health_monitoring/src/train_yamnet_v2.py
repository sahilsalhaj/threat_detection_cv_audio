#!/usr/bin/env python3
"""
train_yamnet_v2.py

Improved training pipeline that:
 - Computes/caches YAMNet segment embeddings
 - Builds parent-level sequences of segment embeddings (padded)
 - Trains a sequence model (BiLSTM -> Dense head) to predict Healthy (queen_acceptance==2) vs Unhealthy
 - Performs parent-level stratified splits
 - Tunes classification threshold on validation set (maximize F1)
 - Saves models, history, metrics, plots into results/ and models/

Usage:
  python src/train_yamnet_v2.py --data_dir training_data --csv all_data_updated.csv

Notes:
 - Requires librosa, tensorflow, tensorflow_hub, numpy, pandas, scikit-learn, matplotlib, seaborn
 - Script auto-creates results/ and models/
"""
import os
import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score
)

# -------------------------
# Utilities
# -------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -------------------------
# Embedding extraction + caching
# -------------------------
def load_or_compute_segment_embeddings(data_dir, yamnet_handle, cache_path, sr=16000, force_recompute=False):
    """
    Computes or loads a cache mapping segment_filename -> 1024-d embedding (avg across frames).
    Cache saved to cache_path (.npz with arrays 'names' and 'embeddings').
    Returns: dict {filename: embedding}
    """
    if os.path.exists(cache_path) and not force_recompute:
        log(f"Loading cached segment embeddings from {cache_path}")
        arr = np.load(cache_path, allow_pickle=True)
        names = arr["names"].tolist()
        embs = arr["embeddings"]
        mapping = {n: embs[i] for i, n in enumerate(names)}
        return mapping

    log("Loading YAMNet model from TF Hub (this may take a moment)...")
    yamnet = hub.load(yamnet_handle)
    log("YAMNet loaded.")

    # collect wav segment files
    wav_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".wav")])
    log(f"Found {len(wav_files)} wav segment files. Computing embeddings...")

    mapping = {}
    failures = 0
    for i, fname in enumerate(wav_files):
        path = os.path.join(data_dir, fname)
        try:
            # use soundfile for robust reading then resample if needed
            audio, file_sr = sf.read(path, dtype='float32')
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if file_sr != sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            audio = audio.astype(np.float32)
            # yamnet expects float waveform 1D
            scores, embeddings, spectrogram = yamnet(audio)
            # embeddings: (n_frames, 1024)
            emb_avg = tf.reduce_mean(embeddings, axis=0).numpy()
            mapping[fname] = emb_avg
        except Exception as e:
            failures += 1
            if i % 200 == 0:
                log(f"Failed at {i} ({fname}): {e}")
            # skip and continue
            continue
        if (i+1) % 200 == 0:
            log(f"Processed {i+1}/{len(wav_files)} segments")
    log(f"Done computing embeddings. Failures: {failures}")

    # save cache
    names = list(mapping.keys())
    embs = np.stack([mapping[n] for n in names], axis=0) if names else np.zeros((0,1024), dtype=np.float32)
    np.savez_compressed(cache_path, names=np.array(names, dtype=object), embeddings=embs)
    log(f"Saved segment embeddings cache to {cache_path} (n={len(names)})")

    return mapping

# -------------------------
# Build parent sequence dataset
# -------------------------
def build_parent_sequences(mapping_segment_embs, csv_df, data_dir, max_segments_cap=None):
    """
    mapping_segment_embs: dict filename -> 1024 vector
    csv_df: dataframe with columns 'file name' and 'queen acceptance'
    Returns:
      parents: list of parent names (.raw)
      X_seq: np.array shape (n_parents, max_segments, 1024)
      lengths: np.array (n_parents,) number of non-padded segments
      y: np.array (n_parents,) binary labels (1=healthy)
      file_lists: list of lists of actual segment filenames for each parent (for saving/demo)
    """
    # create binary label
    if "queen acceptance" not in csv_df.columns:
        raise ValueError("CSV must contain 'queen acceptance' column")
    csv_df = csv_df.copy()
    csv_df["health"] = (csv_df["queen acceptance"] == 2).astype(int)

    # Build mapping parent -> list of segment files found
    parent_to_segments = defaultdict(list)
    # segments in mapping are just basename filenames
    available_segments = set(mapping_segment_embs.keys())
    for seg in available_segments:
        if "__segment" not in seg:
            continue
        parent = seg.split("__segment")[0] + ".raw"
        parent_to_segments[parent].append(seg)

    parents = []
    y = []
    file_lists = []
    # use csv rows as authoritative list of parents (even if some have zero segments)
    for _, row in csv_df.iterrows():
        parent = row["file name"]
        if parent not in parent_to_segments:
            # no segments available for this parent -> skip
            continue
        segs = sorted(parent_to_segments[parent])
        parents.append(parent)
        y.append(int(row["health"]))
        file_lists.append(segs)

    if not parents:
        raise RuntimeError("No parents with available segments found.")

    # determine max_segments
    lengths = np.array([len(lst) for lst in file_lists], dtype=int)
    max_segments = int(lengths.max())
    if max_segments_cap is not None:
        max_segments = min(max_segments, int(max_segments_cap))
    log(f"Parent counts: {len(parents)} | max segments (capped): {max_segments}")

    # build 3D array (pad with zeros)
    n = len(parents)
    X = np.zeros((n, max_segments, 1024), dtype=np.float32)
    lengths_out = np.zeros((n,), dtype=int)

    for i, seg_list in enumerate(file_lists):
        # optionally cap number of segments: keep earliest segments if too many
        if len(seg_list) > max_segments:
            seg_list = seg_list[:max_segments]
        for j, seg in enumerate(seg_list):
            if seg in mapping_segment_embs:
                X[i, j, :] = mapping_segment_embs[seg]
                lengths_out[i] += 1

    return parents, X, lengths_out, np.array(y, dtype=int), file_lists

# -------------------------
# Plot helpers
# -------------------------
def plot_history(history, out_dir, prefix="train"):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(8,4))
    plt.plot(epochs, hist["loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_loss_curve.png")
    plt.savefig(p); plt.close()

    plt.figure(figsize=(8,4))
    acc_key = "accuracy" if "accuracy" in hist else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"
    plt.plot(epochs, hist.get(acc_key, []), label="train_acc")
    plt.plot(epochs, hist.get(val_acc_key, []), label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_accuracy_curve.png")
    plt.savefig(p); plt.close()

def plot_confusion_matrix(y_true, y_pred, out_dir, prefix="test"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    p = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(p); plt.close()

def plot_roc_pr(y_true, y_scores, out_dir, prefix="test"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curve.png")); plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curve.png")); plt.close()
    return roc_auc, ap

# -------------------------
# Model (sequence -> binary)
# -------------------------
def make_sequence_model(max_segments, embedding_dim=1024, lstm_units=128, dropout=0.3):
    """
    Input: (batch, max_segments, embedding_dim)
    Masking on zero rows, BiLSTM, Dense head.
    """
    inp = tf.keras.layers.Input(shape=(max_segments, embedding_dim), name="segment_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=False))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# -------------------------
# Main pipeline
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--csv", type=str, default="all_data_updated.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--yamnet_handle", type=str, default="https://tfhub.dev/google/yamnet/1")
    parser.add_argument("--cache_name", type=str, default="results/segment_embeddings_cache_v2.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_segments_cap", type=int, default=12,
                        help="cap on number of segments per parent (keeps memory stable)")
    parser.add_argument("--force_recompute_cache", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = args.data_dir
    csv_path = os.path.join(data_dir, args.csv)
    ensure_dirs(args.results_dir, args.models_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_path = args.cache_name
    if not os.path.isabs(cache_path):
        cache_path = os.path.join(args.results_dir, os.path.basename(cache_path))

    # 1) compute or load segment embeddings
    mapping = load_or_compute_segment_embeddings(data_dir, args.yamnet_handle, cache_path, force_recompute=args.force_recompute_cache)

    # 2) build parent sequences + labels
    parents, X_seq, lengths, y, file_lists = build_parent_sequences(mapping, df, data_dir, max_segments_cap=args.max_segments_cap)
    log(f"Built parent dataset: N_parents={len(parents)}, sequence_shape={X_seq.shape}")

    # 3) parent-level stratified split: test first, then train/val
    # Use simple train_test_split on parents (they are few) with stratify
    parent_indices = np.arange(len(parents))
    X_par = X_seq
    y_par = y
    idx_temp, idx_test = train_test_split(parent_indices, test_size=args.test_size, stratify=y_par, random_state=args.seed)
    # now split temp into train/val
    val_frac_of_temp = args.val_size / (1.0 - args.test_size)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_frac_of_temp, stratify=y_par[idx_temp], random_state=args.seed)

    X_train, y_train = X_par[idx_train], y_par[idx_train]
    X_val, y_val = X_par[idx_val], y_par[idx_val]
    X_test, y_test = X_par[idx_test], y_par[idx_test]

    fn_train = [parents[i] for i in idx_train]
    fn_val = [parents[i] for i in idx_val]
    fn_test = [parents[i] for i in idx_test]

    log(f"Split sizes — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")
    # save small CSV of which parents in which split
    splits_df = pd.DataFrame({
        "parent": parents,
        "label": y_par,
    })
    splits_df["split"] = ""
    splits_df.loc[idx_train, "split"] = "train"
    splits_df.loc[idx_val, "split"] = "val"
    splits_df.loc[idx_test, "split"] = "test"
    splits_df.to_csv(os.path.join(args.results_dir, f"splits_{timestamp}.csv"), index=False)

    # 4) class weights from train
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, class_weights)}
    log(f"Class weights: {cw}")

    # 5) build model
    max_segments = X_seq.shape[1]
    model = make_sequence_model(max_segments, embedding_dim=1024, lstm_units=128, dropout=0.35)
    model.summary(print_fn=lambda s: log(s))

    # callbacks
    ckpt_path = os.path.join(args.models_dir, f"yamnet_v2_best_{timestamp}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    # 6) train
    log("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=cw,
        callbacks=callbacks,
        shuffle=True,
        verbose=2
    )

    # save final model
    final_model_path = os.path.join(args.models_dir, f"yamnet_v2_final_{timestamp}.keras")
    model.save(final_model_path)
    log(f"Saved final model to: {final_model_path}")
    log(f"Best checkpoint: {ckpt_path}")

    # 7) Evaluate & threshold tune on validation
    log("Predicting on validation set for threshold tuning...")
    val_scores = model.predict(X_val).ravel()
    best_thr = 0.5
    best_f1 = -1.0
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)
    # thresholds length = len(precisions)-1
    for t in np.linspace(0.01, 0.99, 99):
        preds = (val_scores >= t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1 = f; best_thr = t
    log(f"Chosen threshold on val: {best_thr:.3f} (F1={best_f1:.4f})")

    # 8) Test evaluation with best_thr
    log("Predicting on test set...")
    test_scores = model.predict(X_test).ravel()
    test_preds = (test_scores >= best_thr).astype(int)

    # metrics
    roc = roc_auc_score(y_test, test_scores) if len(np.unique(y_test)) > 1 else float("nan")
    ap = average_precision_score(y_test, test_scores) if len(np.unique(y_test)) > 1 else float("nan")
    cls_report = classification_report(y_test, test_preds, output_dict=True)
    cm = confusion_matrix(y_test, test_preds).tolist()

    # save artifacts
    ensure_dirs(args.results_dir, args.models_dir)
    metrics = {
        "timestamp": timestamp,
        "n_parents_total": len(parents),
        "train_parents": int(len(idx_train)),
        "val_parents": int(len(idx_val)),
        "test_parents": int(len(idx_test)),
        "best_threshold": float(best_thr),
        "f1_at_best_threshold_val": float(best_f1),
        "roc_auc_test": float(roc),
        "average_precision_test": float(ap),
        "confusion_matrix": cm,
        "model_final_path": final_model_path,
        "best_ckpt_path": ckpt_path
    }
    metrics_path = os.path.join(args.results_dir, f"metrics_summary_v2_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved metrics summary to {metrics_path}")

    # save classification report CSV
    cls_df = pd.DataFrame(cls_report).transpose()
    cls_report_path = os.path.join(args.results_dir, f"classification_report_v2_{timestamp}.csv")
    cls_df.to_csv(cls_report_path)
    log(f"Saved classification report to {cls_report_path}")

    # save test predictions per parent
    test_pred_df = pd.DataFrame({
        "parent": [fn_test[i] for i in range(len(fn_test))],
        "y_true": y_test,
        "y_score": test_scores,
        "y_pred": test_preds
    })
    test_pred_path = os.path.join(args.results_dir, f"test_predictions_v2_{timestamp}.csv")
    test_pred_df.to_csv(test_pred_path, index=False)
    log(f"Saved test predictions to {test_pred_path}")

    # plots
    plot_confusion_matrix(y_test, test_preds, args.results_dir, prefix=f"v2_{timestamp}")
    plot_roc_pr(y_test, test_scores, args.results_dir, prefix=f"v2_{timestamp}")
    plot_history(history, args.results_dir, prefix=f"v2_{timestamp}")

    # save embeddings used (train/val/test splits)
    np.savez_compressed(os.path.join(args.results_dir, f"embeddings_parent_splits_v2_{timestamp}.npz"),
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    log("Saved parent split embeddings")

    log("ALL DONE. Artifacts saved in results/ and models/")

if __name__ == "__main__":
    main()
