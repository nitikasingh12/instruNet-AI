import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────
MODEL_PATH   = "models/instrunet_cnn.keras"
SAVE_PATH    = "models/instrunet_cnn_finetuned.keras"
SPEC_PATH    = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"
WEAK_CLASSES = ["cel", "flu", "sax"]
ALL_CLASSES  = sorted(["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"])

EPOCHS        = 15
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4


# ── Normalise ───────────────────────────────────────────────────
def normalise(spec):
    mean, var = np.mean(spec), np.var(spec)
    return (spec - mean) / (np.sqrt(var) + 1e-7)


# ── Load weak class spectrograms ───────────────────────────────
print("Loading weak class spectrograms...")
X, y = [], []

for cls in WEAK_CLASSES:
    cls_path = os.path.join(SPEC_PATH, cls)
    files    = [f for f in os.listdir(cls_path) if f.endswith(".npy")]
    print(f"  {cls}: {len(files)} samples found")
    for f in files:
        spec = np.load(os.path.join(cls_path, f)).astype(np.float32)
        X.append(normalise(spec))
        y.append(ALL_CLASSES.index(cls))

X = np.array(X)
y = np.array(y)

# one-hot encode with full 11-class schema
y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(ALL_CLASSES))

X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples : {len(X_train)}")
print(f"Val   samples : {len(X_val)}")


# ── Load saved model ───────────────────────────────────────────
print("\nLoading saved model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# recompile with low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)


# ── Callbacks ──────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )
]


# ── Fine-tune ──────────────────────────────────────────────────
print("\nFine-tuning on weak classes (cel, flu, sax)...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print(f"\nDone! Fine-tuned model saved at: {SAVE_PATH}")
print("Run test.py with instrunet_cnn_finetuned.keras to check improvements!")