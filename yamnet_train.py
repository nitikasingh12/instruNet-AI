import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATASET_PATH    = "C:/Users/NITIKA KUMARI/instrunet-ai/data/raw/IRMAS-TrainingData"
MODEL_SAVE_PATH = "models/yamnet_instrunet.keras"
YAMNET_URL      = "https://tfhub.dev/google/yamnet/1"

SAMPLE_RATE   = 16000   # YAMNet requires 16kHz
DURATION      = 3
N_SAMPLES     = SAMPLE_RATE * DURATION
EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
THRESHOLD     = 0.2

CLASS_NAMES = sorted(["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"])
CLASS_FULL  = {
    "cel": "Cello",           "cla": "Clarinet",
    "flu": "Flute",           "gac": "Acoustic Guitar",
    "gel": "Electric Guitar", "org": "Organ",
    "pia": "Piano",           "sax": "Saxophone",
    "tru": "Trumpet",         "vio": "Violin",
    "voi": "Voice"
}

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -------------------------------------------------------
# PARSE MULTI-HOT LABELS FROM FILENAME
# -------------------------------------------------------
def parse_labels(filename):
    label_vector = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for idx, cls in enumerate(CLASS_NAMES):
        if f"[{cls}]" in filename:
            label_vector[idx] = 1.0
    # fallback — use folder name if no tags in filename
    if label_vector.sum() == 0:
        for idx, cls in enumerate(CLASS_NAMES):
            if cls in filename:
                label_vector[idx] = 1.0
    return label_vector


# -------------------------------------------------------
# LOAD + PREPROCESS AUDIO FOR YAMNET
# YAMNet needs: mono, 16kHz, float32, range [-1, 1]
# -------------------------------------------------------
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y, _  = librosa.effects.trim(y, top_db=30)

    if len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    else:
        y = np.pad(y, (0, N_SAMPLES - len(y)))

    # normalize to [-1, 1]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y.astype(np.float32)


# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------
print("\nLoading dataset...")
X, y = [], []

for cls in CLASS_NAMES:
    cls_dir = os.path.join(DATASET_PATH, cls)
    if not os.path.exists(cls_dir):
        print(f"WARNING: {cls_dir} not found")
        continue
    files = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]
    print(f"  {cls}: {len(files)} files")
    for fname in files:
        try:
            audio  = load_audio(os.path.join(cls_dir, fname))
            labels = parse_labels(fname)
            X.append(audio)
            y.append(labels)
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

X = np.array(X)   # (N, 48000)
y = np.array(y)   # (N, 11)

print(f"\nTotal loaded: {len(X)} samples")


# -------------------------------------------------------
# TRAIN / VAL SPLIT
# -------------------------------------------------------
split_key = np.argmax(y, axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=split_key,
    random_state=42
)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")


# -------------------------------------------------------
# BUILD YAMNET MODEL
# -------------------------------------------------------
print("\nLoading YAMNet from TensorFlow Hub...")

yamnet_model = hub.load(YAMNET_URL)

# extract embeddings function
@tf.function
def get_embeddings(audio):
    _, embeddings, _ = yamnet_model(audio)
    return tf.reduce_mean(embeddings, axis=0)   # (1024,)

print("Extracting embeddings — this may take a few minutes...")

def extract_embeddings(audio_array):
    embeddings = []
    for i, audio in enumerate(audio_array):
        emb = get_embeddings(audio).numpy()
        embeddings.append(emb)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(audio_array)} done")
    return np.array(embeddings)

X_train_emb = extract_embeddings(X_train)   # (N_train, 1024)
X_val_emb   = extract_embeddings(X_val)     # (N_val, 1024)

print(f"\nEmbedding shape: {X_train_emb.shape}")


# -------------------------------------------------------
# CLASSIFICATION HEAD
# -------------------------------------------------------
inputs  = tf.keras.Input(shape=(1024,))
x       = tf.keras.layers.Dense(256, activation='relu')(inputs)
x       = tf.keras.layers.BatchNormalization()(x)
x       = tf.keras.layers.Dropout(0.4)(x)
x       = tf.keras.layers.Dense(128, activation='relu')(x)
x       = tf.keras.layers.BatchNormalization()(x)
x       = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)


# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]


# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------
print("\nTraining classification head...\n")
history = model.fit(
    X_train_emb, y_train,
    validation_data=(X_val_emb, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)


# -------------------------------------------------------
# EVALUATION
# -------------------------------------------------------
print("\nEvaluating...\n")
y_prob = model.predict(X_val_emb, verbose=0)
y_pred = (y_prob >= THRESHOLD).astype(int)

print(classification_report(
    y_val, y_pred,
    target_names=[CLASS_FULL[c] for c in CLASS_NAMES],
    zero_division=0
))

print(f"\nModel saved at: {MODEL_SAVE_PATH}")