# 🎵 InstruNet AI — Music Instrument Recognition System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)

> Automatically detect musical instruments in audio tracks using Convolutional Neural Networks applied to Mel-spectrograms.

---

## 📌 Project Overview

InstruNet AI is a deep learning system that classifies musical instruments from audio files. It uses a custom 4-block CNN trained on Mel-spectrogram representations of audio, capable of detecting 11 instrument classes with segment-based prediction and temporal smoothing.

---

## 🎯 Features

- 🎵 Upload audio files (WAV, MP3, OGG, FLAC)
- 📊 Real-time waveform and spectrogram visualization
- 🔍 Segment-based instrument detection with confidence scores
- 📈 Instrument timeline graph showing activity over time
- 🔬 Advanced per-segment breakdown
- 📥 Export results as JSON or TXT report
- 🌐 Interactive Streamlit web interface

---

## 🎸 Detectable Instruments

| Code | Instrument      |
|------|----------------|
| cel  | Cello          |
| cla  | Clarinet       |
| flu  | Flute          |
| gac  | Acoustic Guitar|
| gel  | Electric Guitar|
| org  | Organ          |
| pia  | Piano          |
| sax  | Saxophone      |
| tru  | Trumpet        |
| vio  | Violin         |
| voi  | Voice          |

---

## 🏗️ System Architecture
```
Audio Input
    ↓
Preprocessing
(Load → Resample → Mono → Trim → Normalize → Segment)
    ↓
Feature Extraction
(Mel-spectrogram + Delta + Delta² → 128×128×3)
    ↓
CNN Model
(4-block CNN → GlobalAvgPool → Dense → Sigmoid)
    ↓
Post-processing
(Segment averaging → Temporal smoothing → Threshold)
    ↓
Output
(Detected instruments + Confidence + Timeline + Report)
```

---

## 🧠 Model Architecture
```
Input: (128, 128, 3)
    ↓
Block 1: Conv2D(32) × 2 + BatchNorm + ReLU + MaxPool + Dropout(0.1)
    ↓
Block 2: Conv2D(64) × 2 + BatchNorm + ReLU + MaxPool + Dropout(0.1)
    ↓
Block 3: Conv2D(128) × 2 + BatchNorm + ReLU + MaxPool + Dropout(0.3)
    ↓
Block 4: Conv2D(256) × 2 + BatchNorm + ReLU + MaxPool + Dropout(0.4)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128) + BatchNorm + Dropout(0.5)
    ↓
Dense(11, activation='sigmoid')
```

---

## 📊 Model Performance

| Metric            | Score  |
|-------------------|--------|
| Overall Accuracy  | 66.2%  |
| Macro F1 Score    | 0.64   |
| Macro AUC         | 0.923  |

### Per-class F1 Scores

| Instrument      | F1    | AUC   |
|----------------|-------|-------|
| Voice          | 0.90  | 0.984 |
| Piano          | 0.82  | 0.951 |
| Organ          | 0.77  | 0.974 |
| Acoustic Guitar| 0.70  | 0.979 |
| Trumpet        | 0.70  | 0.951 |
| Electric Guitar| 0.60  | 0.911 |
| Clarinet       | 0.54  | 0.767 |
| Saxophone      | 0.50  | 0.884 |
| Cello          | 0.49  | 0.942 |
| Violin         | 0.51  | 0.881 |
| Flute          | 0.46  | 0.939 |

### Optimizer Comparison (10 epochs)

| Optimizer | Val Accuracy | Val Loss |
|-----------|-------------|----------|
| SGD       | 0.5585      | 1.6149   |
| RMSprop   | 0.5899      | 1.5276   |
| Adam      | 0.5257      | 1.7068   |

> Adam was used for full training (80 epochs) achieving best long-term accuracy.

---

## 🛠️ Technology Stack

| Category        | Technology              |
|----------------|------------------------|
| Language        | Python 3.10            |
| Deep Learning   | TensorFlow / Keras     |
| Audio Processing| Librosa                |
| Web Interface   | Streamlit              |
| Visualization   | Matplotlib, Seaborn    |
| Dataset         | IRMAS                  |

---

## 📁 Project Structure
```
instrunet-ai/
├── src/
│   ├── dataset_builder.py      # Data pipeline + augmentation
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training script
│   ├── predict.py              # Single file prediction
│   ├── spectrogram_generator.py# Audio → .npy preprocessing
│   ├── data_loader.py          # Dataset loader
│   ├── finetune.py             # Fine-tuning script
│   ├── roc_curve.py            # ROC curve generation
│   └── optimizer_compare.py    # Optimizer comparison
├── app.py                      # Streamlit web application
├── models/                     # Saved model files
├── outputs/                    # Plots and reports
├── data/
│   ├── raw/                    # Raw audio files
│   ├── processed/              # Preprocessed audio
│   └── spectrograms_npy/       # Extracted features
├── requirements.txt            # Dependencies
└── README.md
```

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/instrunet-ai.git
cd instrunet-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the web app
```bash
streamlit run app.py
```

### Predict on a single audio file
```bash
python src/predict.py "path/to/audio.wav"
```

### Train the model
```bash
python src/train.py
```

### Generate ROC curves
```bash
python src/roc_curve.py
```

### Compare optimizers
```bash
python src/optimizer_compare.py
```

---

## 📦 Requirements
```
tensorflow>=2.21
streamlit
librosa
numpy
scikit-learn
matplotlib
seaborn
pandas
```

---

## 📊 Dataset

**IRMAS** (Instrument Recognition in Musical Audio Signals)
- 11 instrument classes
- ~6,700 audio files
- 3-second segments
- Multi-label annotations

> Download from: https://zenodo.org/record/1290750

---

## 🔬 Audio Preprocessing Pipeline
```
Load Audio → Resample (22050 Hz) → Convert to Mono
    → Trim Silence → Normalize → Segment (3s)
    → Mel-spectrogram (128 bins)
    → Delta features (order 1 & 2)
    → Stack channels (128×128×3)
    → Z-score normalization
    → Save as .npy
```

---

## 🎛️ Training Configuration

| Parameter      | Value                    |
|---------------|--------------------------|
| Epochs        | 80 (early stopping)      |
| Batch size    | 32                       |
| Learning rate | 0.0003 (Adam)            |
| Loss          | Binary Crossentropy      |
| Augmentation  | Noise, time/freq masking,|
|               | time shift, mixup (α=0.2)|
| Input shape   | (128, 128, 3)            |
| Output        | Sigmoid (multi-label)    |

---

## 📈 Segmentation Pipeline

For long audio files:
1. Split audio into 3-second segments
2. Predict each segment independently
3. Apply temporal smoothing (window=3)
4. Average predictions across segments
5. Apply threshold (default: 0.30)
6. Return detected instruments

---

## 🤝 Acknowledgements

- IRMAS Dataset — Music Technology Group, Universitat Pompeu Fabra
- Librosa — Audio analysis library
- TensorFlow/Keras — Deep learning framework
- Streamlit — Web app framework

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Nitika Kumari**
CNN-Based Music Instrument Recognition System
