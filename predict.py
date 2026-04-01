
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os
import json

# ── Config ─────────────────────────────────────────────────────
MODEL_PATH  = "C:/Users/NITIKA KUMARI/instrunet-ai/models/instrunet_cnn_v3.keras"
ALL_CLASSES = sorted(["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"])
CLASS_NAMES = {
    "cel": "Cello",           "cla": "Clarinet",
    "flu": "Flute",           "gac": "Acoustic Guitar",
    "gel": "Electric Guitar", "org": "Organ",
    "pia": "Piano",           "sax": "Saxophone",
    "tru": "Trumpet",         "vio": "Violin",
    "voi": "Voice"
}

SAMPLE_RATE    = 22050
SEGMENT_DURATION = 3         # seconds per segment
N_SAMPLES      = SAMPLE_RATE * SEGMENT_DURATION
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048
TARGET_FRAMES  = 128
FMAX           = SAMPLE_RATE // 2
THRESHOLD      = 0.3         # detection threshold
SMOOTH_WEIGHT  = 0.8         # smoothing factor (0=no smooth, 1=full smooth)


# ── Normalise ───────────────────────────────────────────────────
def normalise(spec):
    mean, var = np.mean(spec), np.var(spec)
    return (spec - mean) / (np.sqrt(var) + 1e-7)


# ── Extract features from one segment ──────────────────────────
def extract_segment_features(segment):
    mel     = librosa.feature.melspectrogram(
        y=segment, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    delta1  = librosa.feature.delta(log_mel)
    delta2  = librosa.feature.delta(log_mel, order=2)

    features = np.stack([log_mel, delta1, delta2], axis=-1)

    frames = features.shape[1]
    if frames > TARGET_FRAMES:
        features = features[:, :TARGET_FRAMES, :]
    else:
        features = np.pad(features, ((0,0),(0,TARGET_FRAMES-frames),(0,0)))

    return normalise(features).astype(np.float32)


# ── Segment audio into chunks ───────────────────────────────────
def segment_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y, _  = librosa.effects.trim(y, top_db=30)

    # pad if shorter than one segment
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))

    # split into fixed segments
    segments = []
    num_segments = len(y) // N_SAMPLES
    for i in range(num_segments):
        segment = y[i * N_SAMPLES : (i + 1) * N_SAMPLES]
        segments.append(segment)

    # include last partial segment if > 1 sec
    remainder = y[num_segments * N_SAMPLES:]
    if len(remainder) >= SAMPLE_RATE:
        remainder = np.pad(remainder, (0, N_SAMPLES - len(remainder)))
        segments.append(remainder)

    print(f"  Audio duration : {len(y)/SAMPLE_RATE:.1f}s")
    print(f"  Segments found : {len(segments)}")
    return segments


# ── Apply smoothing across segment predictions ─────────────────
def smooth_predictions(segment_probs):
    smoothed = segment_probs[0].copy()
    for i in range(1, len(segment_probs)):
        smoothed = SMOOTH_WEIGHT * smoothed + (1 - SMOOTH_WEIGHT) * segment_probs[i]
    return smoothed


# ── Main predict function ───────────────────────────────────────
def predict(file_path):
    print(f"\n{'─'*50}")
    print(f"  File : {os.path.basename(file_path)}")
    print(f"{'─'*50}")

    # load model once
    model    = load_model(MODEL_PATH, compile=False)
    segments = segment_audio(file_path)

    # predict on each segment
    segment_probs = []
    for i, segment in enumerate(segments):
        features = extract_segment_features(segment)
        features = np.expand_dims(features, axis=0)
        probs    = model.predict(features, verbose=0)[0]
        segment_probs.append(probs)
        print(f"  Segment {i+1}/{len(segments)} → {CLASS_NAMES[ALL_CLASSES[np.argmax(probs)]]} ({probs.max()*100:.1f}%)")

    segment_probs = np.array(segment_probs)   # (num_segments, 11)

    # ── Raw average (all segments averaged equally)
    raw_avg = np.mean(segment_probs, axis=0)

    # ── Smoothed average (exponential smoothing)
    smoothed_avg = smooth_predictions(segment_probs)

    # ── Compare raw vs smoothed
    print(f"\n{'─'*50}")
    print(f"  Raw avg vs Smoothed — top predictions")
    print(f"{'─'*50}")
    print(f"  {'Instrument':<20} {'Raw':>8} {'Smoothed':>10}")
    print(f"  {'─'*38}")
    for idx in np.argsort(raw_avg)[::-1][:5]:
        name     = CLASS_NAMES[ALL_CLASSES[idx]]
        raw_conf = raw_avg[idx] * 100
        smo_conf = smoothed_avg[idx] * 100
        print(f"  {name:<20} {raw_conf:>7.1f}%  {smo_conf:>8.1f}%")

    # ── Final prediction using smoothed average + threshold
    print(f"\n{'─'*50}")
    print(f"  Final Detected Instruments (threshold: {int(THRESHOLD*100)}%)")
    print(f"{'─'*50}")

    detected = [(i, smoothed_avg[i]) for i in range(11) if smoothed_avg[i] >= THRESHOLD]
    detected = sorted(detected, key=lambda x: x[1], reverse=True)

    if not detected:
        detected = [(np.argmax(smoothed_avg), smoothed_avg[np.argmax(smoothed_avg)])]

    results = {}
    for idx, prob in detected:
        label = ALL_CLASSES[idx]
        name  = CLASS_NAMES[label]
        conf  = prob * 100
        bar   = "█" * int(conf / 5)
        print(f"  {name:<20} {conf:5.1f}%  {bar}")
        results[name] = round(float(conf), 2)

    print(f"{'─'*50}")
    print(f"  Detected {len(detected)} instrument(s)")
    print(f"{'─'*50}\n")

    # ── Export JSON report
    report = {
        "file"        : os.path.basename(file_path),
        "segments"    : len(segments),
        "duration_sec": round(len(segments) * SEGMENT_DURATION, 1),
        "threshold"   : THRESHOLD,
        "detected_instruments": results,
        "all_probabilities": {
            CLASS_NAMES[ALL_CLASSES[i]]: round(float(smoothed_avg[i]) * 100, 2)
            for i in range(11)
        }
    }

    report_path = os.path.splitext(file_path)[0] + "_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {report_path}\n")

    return report


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstruNet — Segment-based Instrument Classifier")
    parser.add_argument("file", type=str, help="Path to audio file (.wav .mp3 .ogg .flac)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found → {args.file}")
    else:
        predict(args.file)