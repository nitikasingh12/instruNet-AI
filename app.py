import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import tempfile
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide"
)

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH    = "models/instrunet_cnn_v3.keras"
ALL_CLASSES   = sorted(["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"])
CLASS_NAMES   = {
    "cel": "Cello",           "cla": "Clarinet",
    "flu": "Flute",           "gac": "Acoustic Guitar",
    "gel": "Electric Guitar", "org": "Organ",
    "pia": "Piano",           "sax": "Saxophone",
    "tru": "Trumpet",         "vio": "Violin",
    "voi": "Voice"
}
SAMPLE_RATE   = 22050
DURATION      = 3
N_SAMPLES     = SAMPLE_RATE * DURATION
N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 2048
TARGET_FRAMES = 128
FMAX          = SAMPLE_RATE // 2
THRESHOLD     = 0.30
COLORS        = [
    "#4A90D9","#2CA87F","#E09132","#D94A4A","#9B4AD9",
    "#4AD9C8","#D9A84A","#4A6ED9","#D94A9B","#7AD94A","#888780"
]


# ── Load model (cached) ─────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH, compile=False)


# ── Feature extraction ───────────────────────────────────────────
def extract_features(y, sr):
    if len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    else:
        y = np.pad(y, (0, N_SAMPLES - len(y)))

    mel     = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    delta1  = librosa.feature.delta(log_mel)
    delta2  = librosa.feature.delta(log_mel, order=2)
    features = np.stack([log_mel, delta1, delta2], axis=-1)

    if features.shape[1] > TARGET_FRAMES:
        features = features[:, :TARGET_FRAMES, :]
    else:
        pad = TARGET_FRAMES - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad),(0,0)))

    mean, var = np.mean(features), np.var(features)
    return ((features - mean) / (np.sqrt(var) + 1e-7)).astype(np.float32)


# ── Predict single segment ───────────────────────────────────────
def predict_segment(model, y, sr):
    features = extract_features(y, sr)
    features = np.expand_dims(features, axis=0)
    return model.predict(features, verbose=0)[0]


# ── Smooth predictions ───────────────────────────────────────────
def smooth(probs, window=3):
    smoothed = np.copy(probs)
    for i in range(len(probs)):
        start = max(0, i - window // 2)
        end   = min(len(probs), i + window // 2 + 1)
        smoothed[i] = np.mean(probs[start:end], axis=0)
    return smoothed


# ── CSS styling ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #4A90D9;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }
    .instrument-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #4A90D9;
    }
    .detected-badge {
        background: #2CA87F;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .not-detected-badge {
        background: #e0e0e0;
        color: #888;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    .metric-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stProgress > div > div {
        background-color: #4A90D9;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎵 InstruNet AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Music Instrument Recognition · Upload. Analyze. Discover.</p>', unsafe_allow_html=True)

st.divider()

# ── Layout ───────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown('<p class="section-header">Upload Audio</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav","mp3","ogg","flac"],
        help="Supported formats: WAV, MP3, OGG, FLAC"
    )

    if uploaded:
        st.audio(uploaded)
        st.markdown(f"**File:** {uploaded.name}")
        st.markdown(f"**Size:** {uploaded.size / 1024:.1f} KB")

    segment_dur = st.slider("Segment duration (seconds)", 2, 5, 3)
    threshold   = st.slider("Detection threshold", 0.1, 0.6, 0.3, 0.05)
    analyze_btn = st.button("🎯 Analyze Track", type="primary", use_container_width=True)


# ── Analysis ─────────────────────────────────────────────────────
if uploaded and analyze_btn:

    with st.spinner("Loading model..."):
        model = load_model_cached()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Processing audio..."):
        y_full, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        y_full, _  = librosa.effects.trim(y_full, top_db=30)
        duration   = len(y_full) / sr

    seg_samples = int(segment_dur * SAMPLE_RATE)
    segments    = []
    seg_times   = []

    i = 0
    while i + seg_samples <= len(y_full):
        segments.append(y_full[i:i + seg_samples])
        seg_times.append(i / SAMPLE_RATE)
        i += seg_samples

    if not segments:
        segments.append(y_full)
        seg_times.append(0.0)

    with right_col:
        st.markdown('<p class="section-header">Analysis Results</p>', unsafe_allow_html=True)

        # waveform + spectrogram
        tab1, tab2 = st.tabs(["🌊 Waveform", "🎨 Spectrogram"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 2.5))
            librosa.display.waveshow(y_full, sr=sr, ax=ax, color='#4A90D9', alpha=0.8)
            ax.set_title("Audio Waveform", fontsize=11)
            ax.set_xlabel("Time (s)")
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#ffffff')
            st.pyplot(fig)
            plt.close()

        with tab2:
            fig, ax = plt.subplots(figsize=(10, 3))
            mel = librosa.feature.melspectrogram(y=y_full, sr=sr, n_mels=128)
            librosa.display.specshow(
                librosa.power_to_db(mel, ref=np.max),
                sr=sr, hop_length=512, x_axis='time', y_axis='mel',
                ax=ax, cmap='magma'
            )
            ax.set_title("Mel Spectrogram", fontsize=11)
            st.pyplot(fig)
            plt.close()

    # predict all segments
    with st.spinner(f"Analyzing {len(segments)} segments..."):
        seg_probs = []
        progress  = st.progress(0)
        for idx, seg in enumerate(segments):
            probs = predict_segment(model, seg, sr)
            seg_probs.append(probs)
            progress.progress((idx + 1) / len(segments))
        progress.empty()

    seg_probs_arr  = np.array(seg_probs)
    smoothed_probs = smooth(seg_probs_arr)
    avg_probs      = np.mean(smoothed_probs, axis=0)

    # detected instruments
    detected = [(ALL_CLASSES[i], avg_probs[i]) for i in range(11) if avg_probs[i] >= threshold]
    detected = sorted(detected, key=lambda x: x[1], reverse=True)
    if not detected:
        detected = [(ALL_CLASSES[np.argmax(avg_probs)], avg_probs[np.argmax(avg_probs)])]

    with right_col:
        st.divider()

        # ── Results section ─────────────────────────────────────
        st.markdown('<p class="section-header">Detected Instruments</p>', unsafe_allow_html=True)

        det_col, conf_col = st.columns([1, 2])

        with det_col:
            for cls, prob in detected:
                st.markdown(f"""
                <div class="instrument-card">
                    <span class="detected-badge">✓ Detected</span><br>
                    <strong>{CLASS_NAMES[cls]}</strong><br>
                    <span style="color:#4A90D9;font-size:1.2rem;font-weight:700;">{prob*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            not_detected = [cls for cls in ALL_CLASSES if cls not in [d[0] for d in detected]]
            with st.expander("Not detected"):
                for cls in not_detected:
                    st.markdown(f'<span class="not-detected-badge">{CLASS_NAMES[cls]}</span> ', unsafe_allow_html=True)

        with conf_col:
            # confidence bars
            st.markdown("**Confidence scores (all instruments)**")
            sorted_idx = np.argsort(avg_probs)[::-1]
            fig, ax = plt.subplots(figsize=(7, 4))
            bar_colors = ['#2CA87F' if avg_probs[i] >= threshold else '#cccccc' for i in sorted_idx]
            bars = ax.barh(
                [CLASS_NAMES[ALL_CLASSES[i]] for i in sorted_idx],
                [avg_probs[i] * 100 for i in sorted_idx],
                color=bar_colors, height=0.6
            )
            ax.axvline(x=threshold * 100, color='#D94A4A', linestyle='--',
                      linewidth=1.5, label=f'Threshold ({threshold*100:.0f}%)')
            ax.set_xlabel('Confidence (%)')
            ax.set_xlim(0, 100)
            ax.legend(fontsize=9)
            for bar, idx in zip(bars, sorted_idx):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{avg_probs[idx]*100:.1f}%', va='center', fontsize=9)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#ffffff')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

        # ── Timeline graph ───────────────────────────────────────
        st.markdown('<p class="section-header">Instrument Timeline</p>', unsafe_allow_html=True)

        detected_cls = [d[0] for d in detected]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for i, cls in enumerate(detected_cls):
            cls_idx  = ALL_CLASSES.index(cls)
            timeline = smoothed_probs[:, cls_idx] * 100
            color    = COLORS[cls_idx]
            ax.plot(seg_times, timeline, marker='o', markersize=4,
                   linewidth=2, color=color, label=CLASS_NAMES[cls])
            ax.fill_between(seg_times, timeline, alpha=0.1, color=color)

        ax.axhline(y=threshold * 100, color='#D94A4A', linestyle='--',
                  linewidth=1, alpha=0.7, label=f'Threshold ({threshold*100:.0f}%)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Confidence (%)')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ── Advanced view ────────────────────────────────────────
        with st.expander("🔬 Advanced View — Per-segment breakdown"):
            st.markdown("**Prediction per segment**")
            seg_data = {}
            for idx, (seg_prob, t) in enumerate(zip(seg_probs_arr, seg_times)):
                top_cls = ALL_CLASSES[np.argmax(seg_prob)]
                seg_data[f"Segment {idx+1} ({t:.1f}s)"] = {
                    CLASS_NAMES[ALL_CLASSES[i]]: f"{seg_prob[i]*100:.1f}%"
                    for i in np.argsort(seg_prob)[::-1][:3]
                }
                st.markdown(f"**Seg {idx+1}** ({t:.1f}s) → "
                           f"**{CLASS_NAMES[top_cls]}** "
                           f"({seg_prob[np.argmax(seg_prob)]*100:.1f}%)")

        st.divider()

        # ── Export section ───────────────────────────────────────
        st.markdown('<p class="section-header">Export Report</p>', unsafe_allow_html=True)

        report = {
            "file"          : uploaded.name,
            "analyzed_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec"  : round(duration, 2),
            "segments"      : len(segments),
            "threshold"     : threshold,
            "detected_instruments": [
                {"instrument": CLASS_NAMES[cls], "confidence": round(prob * 100, 2)}
                for cls, prob in detected
            ],
            "all_scores": {
                CLASS_NAMES[ALL_CLASSES[i]]: round(float(avg_probs[i]) * 100, 2)
                for i in range(11)
            },
            "timeline": [
                {
                    "segment"   : idx + 1,
                    "time_sec"  : round(t, 2),
                    "top_instrument": CLASS_NAMES[ALL_CLASSES[np.argmax(p)]],
                    "confidence": round(float(np.max(p)) * 100, 2)
                }
                for idx, (p, t) in enumerate(zip(seg_probs_arr, seg_times))
            ]
        }

        json_str = json.dumps(report, indent=2)

        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                label="📥 Download JSON Report",
                data=json_str,
                file_name=f"{os.path.splitext(uploaded.name)[0]}_report.json",
                mime="application/json",
                use_container_width=True
            )
        with exp2:
            # simple text summary as PDF alternative
            summary = f"""InstruNet AI — Instrument Analysis Report
==========================================
File          : {uploaded.name}
Analyzed at   : {report['analyzed_at']}
Duration      : {report['duration_sec']}s
Segments      : {report['segments']}
Threshold     : {threshold*100:.0f}%

DETECTED INSTRUMENTS
--------------------
"""
            for item in report['detected_instruments']:
                summary += f"  {item['instrument']:<20} {item['confidence']}%\n"

            summary += "\nALL SCORES\n----------\n"
            for name, score in sorted(report['all_scores'].items(), key=lambda x: -x[1]):
                bar = '█' * int(score / 5)
                summary += f"  {name:<20} {score:5.1f}%  {bar}\n"

            st.download_button(
                label="📄 Download TXT Report",
                data=summary,
                file_name=f"{os.path.splitext(uploaded.name)[0]}_report.txt",
                mime="text/plain",
                use_container_width=True
            )

    os.unlink(tmp_path)

elif not uploaded:
    with right_col:
        st.info("👆 Upload an audio file and click **Analyze Track** to get started.")
        st.markdown("""
        **What InstruNet AI can detect:**
        - 🎻 Cello, Violin
        - 🎷 Clarinet, Saxophone, Flute
        - 🎸 Acoustic Guitar, Electric Guitar
        - 🎹 Piano, Organ
        - 🎺 Trumpet
        - 🎤 Voice
        """)