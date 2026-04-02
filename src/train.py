import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_builder import build_dataset
from model import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ── Hyperparameters ────────────────────────────────────────────
EPOCHS          = 80
LEARNING_RATE   = 0.0003
BATCH_SIZE      = 32
DATASET_PATH    = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"
MODEL_SAVE_PATH = "models/instrunet_cnn.keras"
num_classes     = 11
class_names     = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Macro F1 Metric ────────────────────────────────────────────
class MacroF1(tf.keras.metrics.Metric):

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.f1_sum      = self.add_weight(name='f1_sum', initializer='zeros')
        self.count       = self.add_weight(name='count',  initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        f1_scores = []
        for i in range(self.num_classes):
            tp = tf.reduce_sum(tf.cast((y_pred == i) & (y_true == i), tf.float32))
            fp = tf.reduce_sum(tf.cast((y_pred == i) & (y_true != i), tf.float32))
            fn = tf.reduce_sum(tf.cast((y_pred != i) & (y_true == i), tf.float32))
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


# ── Load Dataset ───────────────────────────────────────────────
print("\nLoading dataset...")
train_data, val_data = build_dataset(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    shuffle_buffer=500
)


# ── Compute Class Weights ──────────────────────────────────────
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

train_labels      = _get_train_labels(DATASET_PATH)
class_weights     = compute_class_weight(
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


# ── Build Model ────────────────────────────────────────────────
model = build_model(num_classes)
print("\nModel Architecture:\n")
model.summary()


# ── Compile ────────────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', MacroF1(num_classes, name='macro_f1')]
)


# ── Callbacks ──────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# ── Train ──────────────────────────────────────────────────────
print("\nStarting Training...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr, checkpoint]
)


# ── Plot Accuracy ──────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/accuracy_curve.png')
plt.close()


# ── Plot Loss ──────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/loss_curve.png')
plt.close()


# ── Plot Macro F1 ──────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history.history['macro_f1'],     label='Training Macro F1')
plt.plot(history.history['val_macro_f1'], label='Validation Macro F1')
plt.title('Training vs Validation Macro F1')
plt.xlabel('Epoch')
plt.ylabel('Macro F1')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/f1_curve.png')
plt.close()


# ── Evaluation ─────────────────────────────────────────────────
print("\nEvaluating Model Performance...\n")

y_true, y_pred = [], []
for x, y in val_data:
    preds = model.predict(x, verbose=0)
    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(preds,     axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

overall_accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {round(overall_accuracy, 4)}")

print("\nPer-Instrument Metrics:\n")
print(classification_report(
    y_true, y_pred,
    target_names=class_names,
    zero_division=0
))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_true, y_pred)
print(cm)


# ── Confusion Matrix Plot ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel='Predicted Label',
    ylabel='True Label',
    title='Confusion Matrix'
)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
plt.close()

print(f"\nBest model saved at : {MODEL_SAVE_PATH}")
print("Plots saved in      : outputs/")