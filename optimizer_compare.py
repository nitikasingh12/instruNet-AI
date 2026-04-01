import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from model import build_model

# ── Config ─────────────────────────────────────────────────────
DATASET_PATH = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"
CLASS_NAMES  = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
EPOCHS       = 10
BATCH_SIZE   = 32
os.makedirs("outputs", exist_ok=True)


# ── Load data ───────────────────────────────────────────────────
print("Loading dataset...")
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

tr_paths, val_paths, tr_labels, val_labels = train_test_split(
    file_paths, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)


# ── Generator ───────────────────────────────────────────────────
def make_dataset(paths, labs, batch_size, shuffle=False):
    def generator():
        for path, lab in zip(paths, labs):
            features  = np.load(path).astype(np.float32)
            mean, var = np.mean(features), np.var(features)
            features  = (features - mean) / (np.sqrt(var) + 1e-7)
            yield features, tf.one_hot(lab, depth=11)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(11,),          dtype=tf.float32)
        )
    )
    if shuffle:
        ds = ds.shuffle(500)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(tr_paths,  tr_labels,  BATCH_SIZE, shuffle=True)
val_ds   = make_dataset(val_paths, val_labels, BATCH_SIZE)

print(f"Train: {len(tr_paths)} | Val: {len(val_paths)}")


# ── Train with each optimizer ───────────────────────────────────
optimizers = {
    'SGD'     : tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop' : tf.keras.optimizers.RMSprop(learning_rate=0.0003),
    'Adam'    : tf.keras.optimizers.Adam(learning_rate=0.0003)
}

results = {}

for opt_name, optimizer in optimizers.items():
    print(f"\nTraining with {opt_name}...")

    model = build_model(num_classes=11)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    results[opt_name] = history.history
    final_val_acc  = max(history.history['val_accuracy'])
    final_val_loss = min(history.history['val_loss'])
    print(f"  {opt_name} → Best Val Acc: {final_val_acc:.4f} | Best Val Loss: {final_val_loss:.4f}")


# ── Plot results ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'SGD': '#D94A4A', 'RMSprop': '#E09132', 'Adam': '#2CA87F'}

for opt_name, history in results.items():
    axes[0].plot(history['val_accuracy'], color=colors[opt_name],
                 linewidth=2, label=opt_name)
    axes[1].plot(history['val_loss'],     color=colors[opt_name],
                 linewidth=2, label=opt_name)

axes[0].set_title('Validation Accuracy — Optimizer Comparison', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Loss — Optimizer Comparison', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/optimizer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Summary table ───────────────────────────────────────────────
print("\n── Optimizer Comparison Summary ────────────────")
print(f"  {'Optimizer':<12} {'Best Val Acc':>14} {'Best Val Loss':>14}")
print(f"  {'─'*42}")
for opt_name, history in results.items():
    best_acc  = max(history['val_accuracy'])
    best_loss = min(history['val_loss'])
    print(f"  {opt_name:<12} {best_acc:>14.4f} {best_loss:>14.4f}")
print(f"  {'─'*42}")
print("\nPlot saved → outputs/optimizer_comparison.png")