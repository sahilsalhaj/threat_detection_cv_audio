#!/usr/bin/env python3
"""
train_cnn.py

High-accuracy Mel-Spectrogram CNN + parent aggregation pipeline for beehive health.

Improvements:
 - Stronger CNN backbone
 - Aggressive augmentation
 - tf.data pipeline (class_weight compatible)
 - Parent-level threshold tuning
 - Full artifact logging
 - Demo parent extraction

Run:
    python src/train_cnn.py --data_dir training_data --csv all_data_updated.csv
"""

import os
import argparse
import json
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Correct TensorFlow + Keras imports for TF 2.20
# -------------------------
import tensorflow as tf
import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight



# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def check_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log(f"GPUs available: {len(gpus)} — {gpus}")
    else:
        log("No GPU detected — training will run on CPU.")



# --------------------------------------------------------------
# Audio → Log-Mel Spectrogram
# --------------------------------------------------------------
def make_log_mel(wave, sr=16000, n_mels=128, n_fft=2048, hop_length=512, target_len=128):
    mel = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize 0–1
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)

    # Pad/trim time axis
    if log_mel.shape[1] < target_len:
        pad = target_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pad)))
    else:
        log_mel = log_mel[:, :target_len]

    return log_mel.astype(np.float32)



# --------------------------------------------------------------
# Spectrogram Caching
# --------------------------------------------------------------
def compute_or_load_spectrogram_cache(data_dir, cache_path, sr=16000, n_mels=128, target_len=128, force=False):
    if os.path.exists(cache_path) and not force:
        log(f"Loading spectrogram cache: {cache_path}")
        arr = np.load(cache_path, allow_pickle=True)
        names = arr["names"].tolist()
        specs = arr["specs"]
        return {names[i]: specs[i] for i in range(len(names))}

    log("Computing spectrograms for all wav segments...")
    wavs = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".wav")])

    mapping = {}
    fails = 0

    for i, fname in enumerate(wavs):
        fp = os.path.join(data_dir, fname)
        try:
            audio, sr_file = sf.read(fp, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr_file != sr:
                audio = librosa.resample(audio, orig_sr=sr_file, target_sr=sr)

            mapping[fname] = make_log_mel(audio, sr=sr, n_mels=n_mels, target_len=target_len)

        except Exception as e:
            fails += 1
            continue

        if (i+1) % 500 == 0:
            log(f"Processed {i+1}/{len(wavs)}")

    # Save cache
    names = list(mapping.keys())
    specs = np.stack([mapping[n] for n in names], axis=0)

    np.savez_compressed(cache_path, names=np.array(names, dtype=object), specs=specs)
    log(f"Saved spectrogram cache: {cache_path} (n={len(names)}, fails={fails})")

    return mapping



# --------------------------------------------------------------
# Parent-based splitting (no leakage)
# --------------------------------------------------------------
def build_parent_splits(mapping, csv_df, test_size=0.15, val_size=0.15, seed=1337):
    df = csv_df.copy()
    df["health"] = (df["queen acceptance"] == 2).astype(int)

    parent_to_segments = defaultdict(list)
    for seg in mapping:
        if "__segment" in seg:
            parent = seg.split("__segment")[0] + ".raw"
            parent_to_segments[parent].append(seg)

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

    idx_temp, idx_test = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    val_frac = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_frac, stratify=labels[idx_temp], random_state=seed)

    def collect(ix):
        segs, labs = [], []
        for i in ix:
            p, lab = parents[i], labels[i]
            for s in file_lists[i]:
                if s in mapping:
                    segs.append(s)
                    labs.append(lab)
        return segs, np.array(labs)

    segs_train, y_train = collect(idx_train)
    segs_val, y_val = collect(idx_val)
    segs_test, y_test = collect(idx_test)

    log(f"PARENTS — train:{len(idx_train)}, val:{len(idx_val)}, test:{len(idx_test)}")
    log(f"SEGMENTS — train:{len(segs_train)}, val:{len(segs_val)}, test:{len(segs_test)}")

    return {
        "parents": parents,
        "file_lists": file_lists,
        "idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test,
        "segs_train": segs_train, "y_train": y_train,
        "segs_val": segs_val, "y_val": y_val,
        "segs_test": segs_test, "y_test": y_test
    }



# --------------------------------------------------------------
# Strong CNN model (improved for higher accuracy)
# --------------------------------------------------------------
def make_cnn_model(input_shape=(128,128,1), dropout=0.4):

    inp = layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(2)(x)
        return x

    x = conv_block(inp, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model



# --------------------------------------------------------------
# Augmentation (spec-wise)
# --------------------------------------------------------------
def augment_spec(s):
    s2 = s.copy()

    # time masking
    if np.random.rand() < 0.5:
        t = s2.shape[1]
        w = np.random.randint(2, t//8)
        st = np.random.randint(0, t-w)
        s2[:, st:st+w] = 0

    # freq masking
    if np.random.rand() < 0.5:
        f = s2.shape[0]
        w = np.random.randint(2, f//8)
        st = np.random.randint(0, f-w)
        s2[st:st+w, :] = 0

    # random gain
    if np.random.rand() < 0.5:
        gain = np.random.uniform(0.8, 1.25)
        s2 = np.clip(s2 * gain, 0, 1)

    return s2



# --------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------
def plot_history(history, out_dir, prefix="cnn"):
    hs = history.history
    epochs = range(1, len(hs["loss"]) + 1)

    # loss
    plt.figure()
    plt.plot(epochs, hs["loss"], label="train")
    plt.plot(epochs, hs["val_loss"], label="val")
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"))
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs, hs["accuracy"], label="train")
    plt.plot(epochs, hs["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.savefig(os.path.join(out_dir, f"{prefix}_acc.png"))
    plt.close()


def plot_confusion(y_true, y_pred, out_dir, prefix="cnn"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion.png"))
    plt.close()


def plot_roc_pr(y_true, scores, out_dir, prefix="cnn"):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"))
    plt.close()

    pr, rc, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    plt.figure()
    plt.plot(rc, pr, label=f"AP={ap:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png"))
    plt.close()

    return roc_auc, ap



# --------------------------------------------------------------
# MAIN TRAINING LOGIC
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--csv", default="all_data_updated.csv")
    parser.add_argument("--results_dir", default="results/cnn_results")
    parser.add_argument("--models_dir", default="models/cnn_models")
    parser.add_argument("--cache", default="cnn_spec_cache.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force_cache", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    check_gpu()
    ensure_dirs(args.results_dir, args.models_dir, "demo_testing/cnn_demo")

    # CSV
    csv_path = os.path.join(args.data_dir, args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    # Spectrogram cache path
    cache_path = os.path.join(args.results_dir, args.cache)

    # 1) Load or compute spectrogram cache
    mapping = compute_or_load_spectrogram_cache(
        args.data_dir, cache_path,
        sr=16000, n_mels=args.n_mels, target_len=args.frames,
        force=args.force_cache
    )

    # 2) Build parent-level splits
    splits = build_parent_splits(mapping, df)

    # Convert filenames → arrays
    def build_xy(seg_list, seg_labels):
        X = np.stack([mapping[s] for s in seg_list], axis=0)
        X = X[..., np.newaxis]
        return X.astype(np.float32), seg_labels.astype(np.float32)

    X_train, y_train = build_xy(splits["segs_train"], splits["y_train"])
    X_val, y_val = build_xy(splits["segs_val"], splits["y_val"])
    X_test, y_test = build_xy(splits["segs_test"], splits["y_test"])

    # 3) Compute class weights
    cw_vals = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train), cw_vals)}
    log(f"Class weights: {class_weights}")

    # 4) Build model
    model = make_cnn_model(input_shape=X_train.shape[1:], dropout=0.4)
    model.summary(print_fn=lambda s: log(s))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(args.models_dir, f"cnn_best_{ts}.keras")
    final_path = os.path.join(args.models_dir, f"cnn_final_{ts}.keras")

    cb = [
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    # --------------------------------------------------------------
    # tf.data AUGMENTATION PIPELINE
    # --------------------------------------------------------------

    def tf_augment(X, y):
        # X shape: (batch, 128,128,1)

        def _aug(arr):
            out = np.empty_like(arr)
            for i in range(arr.shape[0]):
                out[i,:,:,0] = augment_spec(arr[i,:,:,0])
            return out.astype(np.float32)

        X = tf.numpy_function(
            func=_aug,
            inp=[X],
            Tout=tf.float32
        )

        # FIX SHAPE (critical!!)
        X.set_shape([None, 128, 128, 1])

        return X, y


    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train))
    train_ds = train_ds.batch(args.batch)
    train_ds = train_ds.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(args.batch)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    log("Training CNN with tf.data pipeline...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=cb,
        verbose=2
    )

    model.save(final_path)
    log(f"Saved final model: {final_path}")

    # --------------------------------------------------------------
    # VALIDATION — parent-level aggregation + threshold tuning
    # --------------------------------------------------------------
    log("Computing parent-level validation scores...")

    val_seg_probs = model.predict(X_val, batch_size=args.batch).ravel()
    seg_to_prob = dict(zip(splits["segs_val"], val_seg_probs))

    parent_val_scores, parent_val_true = [], []

    for idx_p in splits["idx_val"]:
        p = splits["parents"][idx_p]
        segs = splits["file_lists"][idx_p]
        probs = [seg_to_prob[s] for s in segs if s in seg_to_prob]
        if len(probs)==0:
            continue
        parent_val_scores.append(np.mean(probs))
        parent_val_true.append(int(df.loc[df["file name"]==p, "queen acceptance"].iloc[0]==2))

    parent_val_scores = np.array(parent_val_scores)
    parent_val_true = np.array(parent_val_true)

    best_thr, best_f1 = 0.5, -1
    for t in np.linspace(0.01, 0.99, 95):
        preds = (parent_val_scores >= t).astype(int)
        f = f1_score(parent_val_true, preds)
        if f > best_f1:
            best_f1, best_thr = f, t

    log(f"Best threshold from validation parents: {best_thr:.3f}  (F1={best_f1:.4f})")

    # --------------------------------------------------------------
    # TEST — parent-level evaluation
    # --------------------------------------------------------------
    log("Running test evaluation...")

    test_seg_probs = model.predict(X_test, batch_size=args.batch).ravel()
    seg_to_prob_test = dict(zip(splits["segs_test"], test_seg_probs))

    parent_test_scores, parent_test_true, test_parents = [], [], []

    for idx_p in splits["idx_test"]:
        p = splits["parents"][idx_p]
        test_parents.append(p)
        segs = splits["file_lists"][idx_p]

        probs = [seg_to_prob_test[s] for s in segs if s in seg_to_prob_test]
        if len(probs)==0:
            continue

        parent_test_scores.append(np.mean(probs))
        parent_test_true.append(int(df.loc[df["file name"]==p, "queen acceptance"].iloc[0]==2))

    parent_test_scores = np.array(parent_test_scores)
    parent_test_true = np.array(parent_test_true)
    parent_test_preds = (parent_test_scores >= best_thr).astype(int)

    roc_auc = roc_auc_score(parent_test_true, parent_test_scores)
    ap = average_precision_score(parent_test_true, parent_test_scores)
    rep = classification_report(parent_test_true, parent_test_preds, output_dict=True)
    cm = confusion_matrix(parent_test_true, parent_test_preds)

    # --------------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------------
    metrics = {
        "timestamp": ts,
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "best_threshold": float(best_thr),
        "confusion_matrix": cm.tolist(),
        "class_counts": {int(k): int(v) for k, v in Counter(parent_test_true).items()},
        "model_final_path": final_path,
        "best_ckpt_path": ckpt_path
    }

    with open(os.path.join(args.results_dir, f"metrics_cnn_{ts}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(rep).transpose().to_csv(
        os.path.join(args.results_dir, f"classification_report_cnn_{ts}.csv")
    )

    pd.DataFrame({
        "parent": test_parents,
        "true": parent_test_true,
        "score": parent_test_scores,
        "pred": parent_test_preds
    }).to_csv(os.path.join(args.results_dir, f"test_predictions_cnn_{ts}.csv"))

    plot_confusion(parent_test_true, parent_test_preds, args.results_dir, prefix=f"cnn_{ts}")
    plot_roc_pr(parent_test_true, parent_test_scores, args.results_dir, prefix=f"cnn_{ts}")
    plot_history(history, args.results_dir, prefix=f"cnn_{ts}")

    log(f"TEST RESULTS — ROC AUC={roc_auc:.4f}, AP={ap:.4f}")

    # --------------------------------------------------------------
    # Create Demo Folder
    # --------------------------------------------------------------
    demo_dir = "demo_testing/cnn_demo"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    healthy = [p for p,l in zip(test_parents, parent_test_true) if l==1]
    unhealthy = [p for p,l in zip(test_parents, parent_test_true) if l==0]

    chosen = []
    if len(healthy)>=2 and len(unhealthy)>=2:
        chosen = healthy[:2] + unhealthy[:2]
    else:
        chosen = (healthy + unhealthy)[:4]

    log(f"Demo parents selected: {chosen}")

    for p in chosen:
        prefix = p.replace(".raw","__segment")
        for f in os.listdir(args.data_dir):
            if f.startswith(prefix) and f.endswith(".wav"):
                shutil.copy(os.path.join(args.data_dir, f), os.path.join(demo_dir, f))

    log("CNN training complete. Artifacts saved.")



if __name__ == "__main__":
    main()
