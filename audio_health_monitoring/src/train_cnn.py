import os
import sys
import glob
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import soundfile as sf
import warnings
import traceback
import matplotlib.pyplot as plt
import absl.logging

#######################################################################
# CLEAN ALL NOISY LOGS
#######################################################################
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)

class DevNull:
    def write(self, _): pass
    def flush(self): pass

sys.stderr = DevNull()

#######################################################################
# LOGGING
#######################################################################
LOG_FILE = "cnn_debug_log.txt"

def log(*msg):
    text = " ".join(str(m) for m in msg)
    print(text)
    with open(LOG_FILE, "a", encoding="utf8") as f:
        f.write(text + "\n")


#######################################################################
# MEL SPECTROGRAM GENERATOR
#######################################################################
def audio_to_mel(path, sr=16000, n_mels=64):
    try:
        audio, sr = librosa.load(path, sr=sr)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            fmax=sr // 2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db.astype("float32")
        return mel_db
    except Exception as e:
        log("ERROR converting to mel:", path, e)
        return None


#######################################################################
# MAIN PIPELINE
#######################################################################
def main():
    log("===== CNN TRAINING STARTED =====")

    base_dir = os.getcwd()
    log("Working directory:", base_dir)

    data_dir = "training_data"
    csv_path = os.path.join(data_dir, "all_data_updated.csv")

    if not os.path.exists(csv_path):
        log("ERROR: CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)
    log("Dataset rows:", len(df))

    if "file name" not in df.columns or "target" not in df.columns:
        log("ERROR: Required columns missing.")
        return

    X = []
    Y = []

    ###################################################################
    # Load all spectrograms
    ###################################################################
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            log(f"Processed {idx}/{len(df)} rows")

        raw_name = row["file name"]  # e.g. "2022-06-05--17-41-01_2.raw"
        target = int(row["target"])
        base = raw_name.replace(".raw", "")

        pattern = os.path.join(data_dir, f"{base}_segment*.wav")
        segment_files = sorted(glob.glob(pattern))

        if len(segment_files) == 0:
            continue

        for seg in segment_files:
            mel = audio_to_mel(seg)
            if mel is None:
                continue

            # CNN expects 2D with channel dimension
            mel_resized = np.expand_dims(mel, axis=-1)
            X.append(mel_resized)
            Y.append(target)

    X = np.array(X)
    Y = np.array(Y)

    log("Loaded spectrograms:", X.shape)
    log("Labels:", Y.shape)

    if len(X) == 0:
        log("ERROR: No spectrograms generated.")
        return

    ###################################################################
    # Build CNN model
    ###################################################################
    log("Building CNN model...")

    input_shape = X.shape[1:]

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    ###################################################################
    # Train model
    ###################################################################
    log("Training CNN…")

    try:
        history = model.fit(
            X, Y,
            validation_split=0.2,
            epochs=20,
            batch_size=32
        )
        log("Training complete.")
    except Exception as e:
        log("ERROR during training:", e)
        traceback.print_exc()
        return

    ###################################################################
    # Save model
    ###################################################################
    os.makedirs("models", exist_ok=True)
    out_path = "models/cnn_classifier.h5"

    try:
        model.save(out_path)
        log("CNN model saved:", out_path)
    except Exception as e:
        log("ERROR saving CNN model:", e)

    log("===== CNN TRAINING FINISHED =====")


#######################################################################
# ENTRY POINT
#######################################################################
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FATAL ERROR:", e)
        traceback.print_exc()
