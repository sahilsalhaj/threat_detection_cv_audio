import os
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

# =========================
# PATHS
# =========================
DEMO_DIR = "demo_audio"
MODEL_PATH = "models/cnn_models/cnn_best_20251213_005707.keras"

# =========================
# AUDIO PARAMS (MUST MATCH TRAINING)
# =========================
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_LEN = 128
THRESHOLD = 0.5

# =========================
# LOG-MEL (IDENTICAL LOGIC)
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
# LOAD MODEL
# =========================
print("Loading CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# EVALUATE DEMO FILES
# =========================
files = sorted(f for f in os.listdir(DEMO_DIR) if f.endswith(".wav"))

print("\nEvaluating demo_audio files:\n")

correct = 0

for f in files:
    path = os.path.join(DEMO_DIR, f)

    # expected label from filename
    expected = 1 if f.startswith("healthy_") else 0

    audio, sr_file = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr_file != SR:
        audio = librosa.resample(audio, orig_sr=sr_file, target_sr=SR)

    spec = make_log_mel(audio)
    X = spec[np.newaxis, ..., np.newaxis]  # (1, 128, 128, 1)

    prob = float(model.predict(X, verbose=0)[0][0])
    pred = int(prob >= THRESHOLD)

    is_correct = pred == expected
    correct += int(is_correct)

    print(f"{f}")
    print(f"  expected : {'healthy' if expected else 'unhealthy'}")
    print(f"  predicted: {'healthy' if pred else 'unhealthy'}")
    print(f"  prob     : {prob:.4f}")
    print(f"  result   : {'✅' if is_correct else '❌'}\n")

print(f"Accuracy on demo set: {correct}/{len(files)}")
