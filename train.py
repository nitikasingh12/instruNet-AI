import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_builder import build_dataset
from model import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# -------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------
EPOCHS          = 80
LEARNING_RATE   = 0.0003
BATCH_SIZE      = 32
DATASET_PATH    = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"
MODEL_SAVE_PATH = "models/instrunet_cnn.keras"  # ← v3 clean slate
THRESHOLD       = 0.2                                # ← sigmoid threshold
num_classes     = 11
class_names     = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -------------------------------------------------------
# MACRO F1 METRIC — updated for multi-label sigmoid output
# -------------------------------------------------------
class MacroF1(tf.keras.metrics.Metric):

    def __init__(self, num_classes, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.threshold   = threshold
        self.f1_sum      = self.add_weight(name='f1_sum', initializer='zeros')
        self.count       = self.add_weight(name='count',  initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)  # ← threshold instead of argmax
        f1_scores = []
        for i in range(self.num_classes):
            tp = tf.reduce_sum(y_pred[:, i] * y_true[:, i])
            fp = tf.reduce_sum(y_pred[:, i] * (1 - y_true[:, i]))
            fn = tf.reduce_sum((1 - y_pred[:, i]) * y_true[:, i])
            precision = tp / (tp + fp + 1e-7)
            recall    = tp / (tp + fn + 1e-7)
            f1        = 2 * precision * recall / (precision + recall + 1e-7)
            f1_scores.append(f1)
        self.f1_sum.assign_add(tf.reduce_mean(f1_scores))
        self.count.assign_add(1.0)

    def result(self):
        return self.f1_sum / self.count

    def reset_state(self):
        self.f1_sum.assign(0.0)
        self.count.assign(0.0)


# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------
print("\nLoading dataset...")
train_data, val_data = build_dataset(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    shuffle_buffer=500
)


# -------------------------------------------------------
# COMPUTE CLASS WEIGHTS
# -------------------------------------------------------
print("\nComputing class weights...")

def _get_train_labels(data_path, validation_split=0.2):
    _class_names = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    file_paths, labels = [], []
    for idx, cls in enumerate(_class_names):
        cls_dir = os.path.join(data_path, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.endswith('.npy'):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    file_paths = np.array(file_paths)
    labels     = np.array(labels)
    _, _, tr_labels, _ = train_test_split(
        file_paths, labels,
        test_size=validation_split,
        stratify=labels,
        random_state=42
    )
    return tr_labels

train_labels = _get_train_labels(DATASET_PATH)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass Weights:")
for i, name in enumerate(class_names):
    print(f"  {name}: {class_weight_dict[i]:.4f}")

steps_per_epoch = len(train_labels) // BATCH_SIZE
print(f"\nSteps per epoch: {steps_per_epoch}")


# -------------------------------------------------------
# BUILD + COMPILE MODEL
# -------------------------------------------------------
model = build_model(num_classes)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),  # ← binary
    metrics=['accuracy', MacroF1(num_classes, threshold=THRESHOLD, name='macro_f1')]
)


# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_macro_f1',       # ← monitor F1 not accuracy
        save_best_only=True,
        mode='max',
        verbose=1
    )
]


# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------
print("\nStarting Training...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight_dict,
    callbacks=callbacks
)


# -------------------------------------------------------
# PLOTS
# -------------------------------------------------------
for metric, title, fname in [
    ('accuracy',  'Accuracy',  'accuracy_curve.png'),
    ('loss',      'Loss',      'loss_curve.png'),
    ('macro_f1',  'Macro F1',  'f1_curve.png'),
]:
    plt.figure(figsize=(10, 4))
    plt.plot(history.history[metric],          label=f'Train {title}')
    plt.plot(history.history[f'val_{metric}'], label=f'Val {title}')
    plt.title(f'Training vs Validation {title}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/{fname}')
    plt.show()


# -------------------------------------------------------
# EVALUATION — threshold based for multi-label
# -------------------------------------------------------
print("\nEvaluating Model Performance...\n")

y_true_all = []
y_pred_all = []

for x, y in val_data:
    preds = model.predict(x, verbose=0)
    y_true_all.append(y.numpy())
    y_pred_all.append((preds >= THRESHOLD).astype(int))  # ← threshold

y_true = np.vstack(y_true_all)
y_pred = np.vstack(y_pred_all)

# per-class report
print("\nPer-Instrument Metrics:\n")
print(classification_report(
    y_true, y_pred,
    target_names=class_names,
    zero_division=0
))

# overall subset accuracy (all labels correct for a sample)
subset_acc = accuracy_score(y_true, y_pred)
print(f"Subset Accuracy (exact match): {subset_acc:.4f}")

print(f"\nBest model saved at : {MODEL_SAVE_PATH}")
print("Plots saved in      : outputs/")