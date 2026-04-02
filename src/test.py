import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

# ── Config ─────────────────────────────────────────────────────
MODEL_PATH   = "models/instrunet_cnn.keras"
DATASET_PATH = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"
CLASS_NAMES  = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
CLASS_FULL   = {
    'cel': 'Cello',          'cla': 'Clarinet',
    'flu': 'Flute',          'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar','org': 'Organ',
    'pia': 'Piano',          'sax': 'Saxophone',
    'tru': 'Trumpet',        'vio': 'Violin',
    'voi': 'Voice'
}

SAMPLE_RATE   = 22050
DURATION      = 3
N_SAMPLES     = SAMPLE_RATE * DURATION
N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 2048
TARGET_FRAMES = 128
FMAX          = SAMPLE_RATE // 2


# ── Normalise ───────────────────────────────────────────────────
def normalise(spec):
    mean, var = np.mean(spec), np.var(spec)
    return (spec - mean) / (np.sqrt(var) + 1e-7)


# ── Extract features from audio file ───────────────────────────
def extract_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y, _  = librosa.effects.trim(y, top_db=30)
    y     = y[:N_SAMPLES] if len(y) > N_SAMPLES else np.pad(y, (0, N_SAMPLES - len(y)))
    mel      = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX
    )
    log_mel  = librosa.power_to_db(mel, ref=np.max)
    features = np.stack([log_mel,
                         librosa.feature.delta(log_mel),
                         librosa.feature.delta(log_mel, order=2)], axis=-1)
    frames   = features.shape[1]
    features = features[:, :TARGET_FRAMES, :] if frames > TARGET_FRAMES else \
               np.pad(features, ((0,0),(0,TARGET_FRAMES-frames),(0,0)))
    return normalise(features).astype(np.float32)


# ── Extract features from .npy file ────────────────────────────
def extract_from_npy(file_path):
    return normalise(np.load(file_path).astype(np.float32))


# ── Load model ──────────────────────────────────────────────────
def get_model():
    print(f"Loading model: {MODEL_PATH}")
    return load_model(MODEL_PATH, compile=False)


# ══════════════════════════════════════════════════════════════
# MODE 1 — Evaluate on full validation set
# ══════════════════════════════════════════════════════════════
def evaluate_validation():
    print("\n" + "="*55)
    print("  MODE 1 — Full Validation Set Evaluation")
    print("="*55)

    file_paths, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(DATASET_PATH, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.endswith('.npy'):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)

    file_paths = np.array(file_paths)
    labels     = np.array(labels)

    _, val_paths, _, val_labels = train_test_split(
        file_paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print(f"Validation samples : {len(val_paths)}")

    model = get_model()

    print("Running predictions...")
    y_true, y_pred = [], []
    for i, (path, label) in enumerate(zip(val_paths, val_labels)):
        features = extract_from_npy(path)
        features = np.expand_dims(features, axis=0)
        probs    = model.predict(features, verbose=0)[0]
        y_true.append(label)
        y_pred.append(np.argmax(probs))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(val_paths)} done")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\nOverall Accuracy : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print("\nPer-Instrument Metrics:\n")
    print(classification_report(
        y_true, y_pred,
        target_names=[CLASS_FULL[c] for c in CLASS_NAMES],
        zero_division=0
    ))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# ══════════════════════════════════════════════════════════════
# MODE 2 — Test on a single audio file
# ══════════════════════════════════════════════════════════════
def test_single_file(file_path):
    print("\n" + "="*55)
    print("  MODE 2 — Single File Prediction")
    print("="*55)

    if not os.path.exists(file_path):
        print(f"Error: File not found → {file_path}")
        return

    ext   = os.path.splitext(file_path)[1].lower()
    model = get_model()

    features = extract_from_npy(file_path) if ext == ".npy" \
               else extract_from_audio(file_path)

    features = np.expand_dims(features, axis=0)
    probs    = model.predict(features, verbose=0)[0]
    top3     = np.argsort(probs)[::-1][:3]

    print(f"\nFile : {os.path.basename(file_path)}")
    print(f"\n── Prediction ──────────────────────────────")
    for i, idx in enumerate(top3):
        name = CLASS_FULL[CLASS_NAMES[idx]]
        conf = probs[idx] * 100
        bar  = "█" * int(conf / 5)
        tag  = " ← TOP" if i == 0 else ""
        print(f"  {name:<20} {conf:5.1f}%  {bar}{tag}")
    print(f"────────────────────────────────────────────")

    top_conf = probs[top3[0]] * 100
    if top_conf >= 70:
        print("  Status : High confidence ✓")
    elif top_conf >= 50:
        print("  Status : Moderate confidence ✓")
    else:
        print("  Status : Low confidence — mixed audio likely")


# ══════════════════════════════════════════════════════════════
# MODE 3 — Batch test on a folder
# ══════════════════════════════════════════════════════════════
def test_batch_folder(folder_path):
    print("\n" + "="*55)
    print("  MODE 3 — Batch Folder Test")
    print("="*55)

    if not os.path.exists(folder_path):
        print(f"Error: Folder not found → {folder_path}")
        return

    supported = [".wav", ".mp3", ".ogg", ".flac", ".npy"]
    files     = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in supported]

    if not files:
        print("No supported audio files found in folder.")
        return

    print(f"Found {len(files)} files\n")
    model = get_model()

    results = []
    for fname in sorted(files):
        file_path = os.path.join(folder_path, fname)
        ext       = os.path.splitext(fname)[1].lower()
        try:
            features = extract_from_npy(file_path) if ext == ".npy" \
                       else extract_from_audio(file_path)
            features  = np.expand_dims(features, axis=0)
            probs     = model.predict(features, verbose=0)[0]
            top_idx   = np.argmax(probs)
            top_label = CLASS_FULL[CLASS_NAMES[top_idx]]
            top_conf  = probs[top_idx] * 100
            results.append((fname, top_label, top_conf))
            print(f"  {fname:<45} → {top_label:<20} {top_conf:.1f}%")
        except Exception as e:
            print(f"  {fname:<45} → ERROR: {e}")

    print(f"\n── Summary ─────────────────────────────────")
    label_counts = Counter([r[1] for r in results])
    for label, count in label_counts.most_common():
        bar = "█" * count
        print(f"  {label:<20} {count:3}x  {bar}")
    print(f"────────────────────────────────────────────")
    print(f"  Total files tested : {len(results)}")
    print(f"  Average confidence : {np.mean([r[2] for r in results]):.1f}%")


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstruNet — Test Script")
    parser.add_argument("--mode",   type=str, default="eval",
                        choices=["eval", "single", "batch"],
                        help="eval=validation set, single=one file, batch=folder")
    parser.add_argument("--file",   type=str, default=None,
                        help="Path to single audio file (for single mode)")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to folder of audio files (for batch mode)")
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_validation()
    elif args.mode == "single":
        if not args.file:
            print("Error: --file required for single mode")
            print("Usage: python src/test.py --mode single --file path/to/audio.wav")
        else:
            test_single_file(args.file)
    elif args.mode == "batch":
        if not args.folder:
            print("Error: --folder required for batch mode")
            print("Usage: python src/test.py --mode batch --folder path/to/folder")
        else:
            test_batch_folder(args.folder)