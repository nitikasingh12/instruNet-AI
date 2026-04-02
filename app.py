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
MODEL_PATH    = "models/instrunet_cnn.keras"
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
COLORS        = [
    "#E75480","#2CA87F","#E09132","#D94A4A","#9B4AD9",
    "#4AD9C8","#D9A84A","#C75470","#D94A9B","#7AD94A","#888780"
]

 
# ── Session state ───────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #121212; color: #e0e0e0; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E75480, #1DB954);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    .subtitle {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
        margin-bottom: 1.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .section-header {
        font-size: 0.85rem;
        font-weight: 700;
        color: #E75480;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #282828;
        padding-bottom: 8px;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    /* Login page */
    .login-card {
        background: #1E1E1E;
        border: 1px solid #282828;
        border-radius: 16px;
        padding: 2.5rem;
        max-width: 420px;
        margin: 3rem auto;
        text-align: center;
    }
    .login-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E75480, #1DB954);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .login-sub {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .welcome-badge {
        background: linear-gradient(135deg, #E75480, #1DB954);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* Instrument cards */
    .instrument-card {
        background: linear-gradient(135deg, #1E1E1E, #282828);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border: 1px solid #282828;
        border-left: 4px solid #E75480;
    }
    .detected-badge {
        background: linear-gradient(135deg, #1a4a2e, #1e5a35);
        color: #1DB954;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: 1px solid #1DB95444;
    }
    .not-detected-badge {
        background: #1a1a1a;
        color: #444;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        border: 1px solid #222;
        display: inline-block;
        margin: 3px;
    }
    .conf-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E75480, #1DB954);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .inst-name {
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 4px 0;
    }

    /* Metric boxes */
    .metric-box {
        background: #1E1E1E;
        border: 1px solid #282828;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #E75480; }
    .metric-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #888;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #282828 !important;
        color: #E75480 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #E75480, #1DB954) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        padding: 0.6rem 1.5rem !important;
    }
    .stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

    /* Download buttons */
    .stDownloadButton > button {
        background: #1E1E1E !important;
        color: #E75480 !important;
        border: 1px solid #E75480 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stDownloadButton > button:hover { background: #E75480 !important; color: white !important; }

    /* Input fields */
    .stTextInput > div > div > input {
        background: #282828 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #E75480 !important;
        box-shadow: 0 0 0 1px #E75480 !important;
    }

    hr { border-color: #282828 !important; }
    .streamlit-expanderHeader {
        background: #1E1E1E !important;
        border-radius: 10px !important;
        color: #888 !important;
        font-size: 0.85rem !important;
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #121212; }
    ::-webkit-scrollbar-thumb { background: #282828; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════
def show_login():

    # Initialize session states
    if "users" not in st.session_state:
        st.session_state.users = {}

    if "page" not in st.session_state:
        st.session_state.page = "login"

    st.markdown("""
    <div style="text-align:center; margin-top: 2rem;">
        <p style="font-size:3rem; margin:0;">🎵</p>
        <p class="login-title">InstruNet AI</p>
        <p class="login-sub">Music Instrument Recognition</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.markdown("""
        <div style="background:#1E1E1E; border:1px solid #282828;
             border-radius:16px; padding:2rem; text-align:center;">
            <p style="font-size:1.2rem; font-weight:700; color:#e0e0e0;">
                Welcome 👇
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Toggle buttons
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                st.session_state.page = "login"
        with c2:
            if st.button("Signup", use_container_width=True):
                st.session_state.page = "signup"

        # ---------------- LOGIN ----------------
        if st.session_state.page == "login":
            with st.form("login_form"):
                email = st.text_input("👤 Email", placeholder="Enter email")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("🎵 Sign In", use_container_width=True)

                if submit:
                    if email in st.session_state.users and st.session_state.users[email] == password:
                        st.session_state.logged_in = True
                        st.session_state.username = email
                        st.success("Login successful ✅")
                        st.rerun()
                    else:
                        st.error("Invalid email or password ❌")

        # ---------------- SIGNUP ----------------
        elif st.session_state.page == "signup":
            with st.form("signup_form"):
                new_email = st.text_input("📧 Email", placeholder="Enter email")
                new_password = st.text_input("🔑 Password", type="password", placeholder="Create password")
                submit = st.form_submit_button("✨ Sign Up", use_container_width=True)

                if submit:
                    if new_email in st.session_state.users:
                        st.warning("User already exists ⚠️")
                    else:
                        st.session_state.users[new_email] = new_password
                        st.success("Account created 🎉 Now login")
                        st.session_state.page = "login"
                        st.rerun()

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

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
    mel      = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX
    )
    log_mel  = librosa.power_to_db(mel, ref=np.max)
    delta1   = librosa.feature.delta(log_mel)
    delta2   = librosa.feature.delta(log_mel, order=2)
    features = np.stack([log_mel, delta1, delta2], axis=-1)
    if features.shape[1] > TARGET_FRAMES:
        features = features[:, :TARGET_FRAMES, :]
    else:
        pad = TARGET_FRAMES - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad),(0,0)))
    mean, var = np.mean(features), np.var(features)
    return ((features - mean) / (np.sqrt(var) + 1e-7)).astype(np.float32)


def predict_segment(model, y, sr):
    features = extract_features(y, sr)
    features = np.expand_dims(features, axis=0)
    return model.predict(features, verbose=0)[0]


def smooth(probs, window=3):
    smoothed = np.copy(probs)
    for i in range(len(probs)):
        start = max(0, i - window // 2)
        end   = min(len(probs), i + window // 2 + 1)
        smoothed[i] = np.mean(probs[start:end], axis=0)
    return smoothed


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32,  np.int64)):   return int(obj)
        return super().default(obj)


def show_main_app():
    # header
    col_logo, col_title, col_user = st.columns([1, 7, 2])
    with col_logo:
        st.markdown("# 🎵")
    with col_title:
        st.markdown('<p class="main-title">InstruNet AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Music Instrument Recognition · Upload. Analyze. Discover.</p>', unsafe_allow_html=True)
    with col_user:
        st.markdown(f"""
        <div style="text-align:right; padding-top:0.8rem;">
            <span class="welcome-badge">👤 {st.session_state.username}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign Out", key="logout"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()

    st.divider()

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<p class="section-header">Upload Audio</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["wav","mp3","ogg","flac"],
            help="Supported: WAV, MP3, OGG, FLAC"
        )
        if uploaded:
            st.audio(uploaded)
            st.markdown(f"**File:** {uploaded.name}")
            st.markdown(f"**Size:** {uploaded.size / 1024:.1f} KB")

        st.markdown('<p class="section-header">Settings</p>', unsafe_allow_html=True)
        segment_dur = st.slider("Segment duration (s)", 2, 5, 3)
        threshold   = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
        analyze_btn = st.button("🎯  Analyze Track", type="primary", use_container_width=True)

        if uploaded and analyze_btn:
            st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)

    # ── Analysis ─────────────────────────────────────────────────
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
        segments, seg_times = [], []
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
            tab1, tab2 = st.tabs(["🌊 Waveform", "🎨 Spectrogram"])
            plt.style.use('dark_background')

            with tab1:
                fig, ax = plt.subplots(figsize=(10, 2.5))
                librosa.display.waveshow(y_full, sr=sr, ax=ax, color='#E75480', alpha=0.8)
                ax.set_title("Audio Waveform", fontsize=11, color='#e0e0e0')
                ax.set_xlabel("Time (s)", color='#888')
                ax.tick_params(colors='#888')
                ax.set_facecolor('#1E1E1E')
                fig.patch.set_facecolor('#121212')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#282828')
                st.pyplot(fig)
                plt.close()

            with tab2:
                fig, ax = plt.subplots(figsize=(10, 3))
                mel = librosa.feature.melspectrogram(y=y_full, sr=sr, n_mels=128)
                img = librosa.display.specshow(
                    librosa.power_to_db(mel, ref=np.max),
                    sr=sr, hop_length=512, x_axis='time', y_axis='mel',
                    ax=ax, cmap='magma'
                )
                plt.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set_title("Mel Spectrogram", fontsize=11, color='#e0e0e0')
                ax.set_facecolor('#1E1E1E')
                fig.patch.set_facecolor('#121212')
                st.pyplot(fig)
                plt.close()

        # predict segments
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

        detected = [(ALL_CLASSES[i], avg_probs[i]) for i in range(11) if avg_probs[i] >= threshold]
        detected = sorted(detected, key=lambda x: x[1], reverse=True)
        if not detected:
            detected = [(ALL_CLASSES[np.argmax(avg_probs)], avg_probs[np.argmax(avg_probs)])]

        # summary metrics
        with left_col:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value">{len(segments)}</div>
                    <div class="metric-label">Segments</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value">{len(detected)}</div>
                    <div class="metric-label">Detected</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-box" style="margin-top:10px">
                <div class="metric-value">{duration:.1f}s</div>
                <div class="metric-label">Duration</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-box" style="margin-top:10px">
                <div class="metric-value">{avg_probs[np.argmax(avg_probs)]*100:.0f}%</div>
                <div class="metric-label">Top confidence</div>
            </div>""", unsafe_allow_html=True)

        with right_col:
            st.divider()

            # detected instruments
            st.markdown('<p class="section-header">Detected Instruments</p>', unsafe_allow_html=True)
            det_col, conf_col = st.columns([1, 2])

            with det_col:
                for cls, prob in detected:
                    st.markdown(f"""
                    <div class="instrument-card">
                        <span class="detected-badge">✓ Detected</span>
                        <p class="inst-name">{CLASS_NAMES[cls]}</p>
                        <span class="conf-value">{prob*100:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

                detected_cls_list = [d[0] for d in detected]
                not_detected = [cls for cls in ALL_CLASSES if cls not in detected_cls_list]
                with st.expander("Not detected"):
                    for cls in not_detected:
                        st.markdown(f'<span class="not-detected-badge">{CLASS_NAMES[cls]}</span>', unsafe_allow_html=True)

            with conf_col:
                st.markdown("**Confidence scores**")
                sorted_idx = np.argsort(avg_probs)[::-1]
                fig, ax    = plt.subplots(figsize=(7, 4))
                bar_colors = ['#E75480' if avg_probs[i] >= threshold else '#1E1E1E' for i in sorted_idx]
                bars       = ax.barh(
                    [CLASS_NAMES[ALL_CLASSES[i]] for i in sorted_idx],
                    [avg_probs[i] * 100 for i in sorted_idx],
                    color=bar_colors, height=0.6
                )
                ax.axvline(x=threshold * 100, color='#1DB954', linestyle='--',
                           linewidth=1.5, label=f'Threshold ({threshold*100:.0f}%)')
                ax.set_xlabel('Confidence (%)', color='#888')
                ax.set_xlim(0, 100)
                ax.legend(fontsize=9)
                ax.tick_params(colors='#888')
                for bar, idx in zip(bars, sorted_idx):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                            f'{avg_probs[idx]*100:.1f}%', va='center', fontsize=9, color='#888')
                ax.set_facecolor('#1E1E1E')
                fig.patch.set_facecolor('#121212')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#282828')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.divider()

            # timeline
            st.markdown('<p class="section-header">Instrument Timeline</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            for cls, _ in detected:
                cls_idx  = ALL_CLASSES.index(cls)
                timeline = smoothed_probs[:, cls_idx] * 100
                color    = COLORS[cls_idx]
                ax.plot(seg_times, timeline, marker='o', markersize=5,
                        linewidth=2.5, color=color, label=CLASS_NAMES[cls])
                ax.fill_between(seg_times, timeline, alpha=0.15, color=color)
            ax.axhline(y=threshold * 100, color='#1DB954', linestyle='--',
                       linewidth=1, alpha=0.7, label=f'Threshold ({threshold*100:.0f}%)')
            ax.set_xlabel('Time (seconds)', color='#888')
            ax.set_ylabel('Confidence (%)', color='#888')
            ax.set_ylim(0, 100)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.15, color='#333')
            ax.tick_params(colors='#888')
            ax.set_facecolor('#1E1E1E')
            fig.patch.set_facecolor('#121212')
            for spine in ax.spines.values():
                spine.set_edgecolor('#282828')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.divider()

            # advanced view
            with st.expander("🔬 Advanced View — Per-segment breakdown"):
                for idx, (seg_prob, t) in enumerate(zip(seg_probs_arr, seg_times)):
                    top_cls  = ALL_CLASSES[np.argmax(seg_prob)]
                    top_conf = seg_prob[np.argmax(seg_prob)] * 100
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.markdown(f"**Seg {idx+1}** ({t:.1f}s)")
                    with col_b:
                        st.markdown(f"→ **{CLASS_NAMES[top_cls]}** `{top_conf:.1f}%`")

            # model reflection
            with st.expander("📊 Model Reflection"):
                st.markdown(f"""
                **Model:** InstruNet CNN (4-block CNN)
                **Task:** Instrument classification (11 classes)
                **Accuracy:** 66.20% | **Macro AUC:** 0.923
                **Dataset:** IRMAS

                > Predictions averaged across {len(segments)} segments
                > for robust final detection.
                """)

            st.divider()

            # export
            st.markdown('<p class="section-header">Export Report</p>', unsafe_allow_html=True)

            report = {
                "file"        : uploaded.name,
                "analyzed_by" : st.session_state.username,
                "analyzed_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": round(float(duration), 2),
                "segments"    : len(segments),
                "threshold"   : float(threshold),
                "detected_instruments": [
                    {"instrument": CLASS_NAMES[cls], "confidence": round(float(prob) * 100, 2)}
                    for cls, prob in detected
                ],
                "all_scores": {
                    CLASS_NAMES[ALL_CLASSES[i]]: round(float(avg_probs[i]) * 100, 2)
                    for i in range(11)
                },
                "timeline": [
                    {
                        "segment"       : idx + 1,
                        "time_sec"      : round(float(t), 2),
                        "top_instrument": CLASS_NAMES[ALL_CLASSES[np.argmax(p)]],
                        "confidence"    : round(float(np.max(p)) * 100, 2)
                    }
                    for idx, (p, t) in enumerate(zip(seg_probs_arr, seg_times))
                ]
            }

            json_str = json.dumps(report, indent=2, cls=NumpyEncoder)
            summary  = f"""InstruNet AI — Instrument Analysis Report
==========================================
File          : {uploaded.name}
Analyzed by   : {st.session_state.username}
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

            exp1, exp2 = st.columns(2)
            with exp1:
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{os.path.splitext(uploaded.name)[0]}_report.json",
                    mime="application/json",
                    use_container_width=True
                )
            with exp2:
                st.download_button(
                    label="📄 Download TXT",
                    data=summary,
                    file_name=f"{os.path.splitext(uploaded.name)[0]}_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        os.unlink(tmp_path)

    elif not uploaded:
        with right_col:
            st.markdown('<p class="section-header">Welcome</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#1E1E1E;border:1px solid #282828;border-radius:12px;
                 padding:2rem;text-align:center;">
                <p style="font-size:3rem;margin:0;">🎵</p>
                <p style="color:#888;font-size:1rem;margin:0.5rem 0;">
                    Welcome back, <strong style="color:#E75480">{st.session_state.username}</strong>!</p>
                <p style="color:#444;font-size:0.85rem;">Upload an audio file to get started</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-header">Detectable Instruments</p>', unsafe_allow_html=True)
            instruments = [
                ("🎻","Cello"),      ("🎷","Clarinet"),   ("🪈","Flute"),
                ("🎸","Ac. Guitar"), ("🎸","El. Guitar"),  ("🎹","Organ"),
                ("🎹","Piano"),      ("🎷","Saxophone"),   ("🎺","Trumpet"),
                ("🎻","Violin"),     ("🎤","Voice"),
            ]
            cols = st.columns(4)
            for i, (emoji, name) in enumerate(instruments):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style="background:#1E1E1E;border:1px solid #282828;
                         border-radius:10px;padding:12px;text-align:center;margin-bottom:8px;">
                        <div style="font-size:1.5rem;">{emoji}</div>
                        <div style="font-size:0.8rem;color:#888;">{name}</div>
                    </div>""", unsafe_allow_html=True)


# ── Router ───────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    show_main_app()