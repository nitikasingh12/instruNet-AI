import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# ── Config ─────────────────────────────────────────────────────
MODEL_PATH   = "C:/Users/NITIKA KUMARI/instrunet-ai/models/instrunet_cnn_v3.keras"
DATASET_PATH = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"
CLASS_NAMES  = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
CLASS_FULL   = {
    'cel':'Cello',           'cla':'Clarinet',
    'flu':'Flute',           'gac':'Acoustic Guitar',
    'gel':'Electric Guitar', 'org':'Organ',
    'pia':'Piano',           'sax':'Saxophone',
    'tru':'Trumpet',         'vio':'Violin',
    'voi':'Voice'
}

os.makedirs("outputs", exist_ok=True)


# ── Load val file paths ─────────────────────────────────────────
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

_, val_paths, _, val_labels = train_test_split(
    file_paths, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(f"Validation samples : {len(val_paths)}")


# ── Load model ──────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)


# ── Run predictions ─────────────────────────────────────────────
print("Running predictions...")
y_true_list = []
y_prob_list = []

for i, (path, label) in enumerate(zip(val_paths, val_labels)):
    features      = np.load(path).astype(np.float32)
    mean, var     = np.mean(features), np.var(features)
    features      = (features - mean) / (np.sqrt(var) + 1e-7)
    features      = np.expand_dims(features, axis=0)
    probs         = model.predict(features, verbose=0)[0]
    y_true_list.append(label)
    y_prob_list.append(probs)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(val_paths)} done")

y_true_raw = np.array(y_true_list)
y_prob     = np.array(y_prob_list)

# binarize for ROC
y_true = label_binarize(y_true_raw, classes=list(range(11)))


# ── Plot ROC curves ─────────────────────────────────────────────
colors = ['#4A90D9','#2CA87F','#E09132','#D94A4A','#9B4AD9',
          '#4AD9C8','#D9A84A','#4A6ED9','#D94A9B','#7AD94A','#888780']

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

all_fpr = []
all_tpr = []
all_auc = []

for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
    roc_auc     = auc(fpr, tpr)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

    ax = axes[i]
    ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0,1],[0,1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_title(CLASS_FULL[cls], fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

# macro average
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(mean_fpr)
for i in range(len(CLASS_NAMES)):
    mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
mean_tpr /= len(CLASS_NAMES)
mean_auc  = auc(mean_fpr, mean_tpr)

ax = axes[11]
for i, cls in enumerate(CLASS_NAMES):
    ax.plot(all_fpr[i], all_tpr[i], color=colors[i],
            linewidth=1, alpha=0.4, label=f'{cls} ({all_auc[i]:.2f})')
ax.plot(mean_fpr, mean_tpr, color='black',
        linewidth=2.5, label=f'Macro avg ({mean_auc:.3f})')
ax.plot([0,1],[0,1], 'k--', linewidth=0.8, alpha=0.5)
ax.set_title('Macro Average ROC', fontsize=11, fontweight='bold')
ax.set_xlabel('False Positive Rate', fontsize=9)
ax.set_ylabel('True Positive Rate', fontsize=9)
ax.legend(loc='lower right', fontsize=7)
ax.grid(True, alpha=0.3)

plt.suptitle('ROC Curves — InstruNet Instrument Classifier',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Print AUC scores ────────────────────────────────────────────
print("\n── ROC AUC Scores ──────────────────────────")
for i, cls in enumerate(CLASS_NAMES):
    bar = '█' * int(all_auc[i] * 20)
    print(f"  {CLASS_FULL[cls]:<20} AUC: {all_auc[i]:.3f}  {bar}")
print(f"\n  Macro Average AUC : {mean_auc:.3f}")
print("────────────────────────────────────────────")
print("\nROC curve saved → outputs/roc_curves.png")import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# ── Config ─────────────────────────────────────────────────────
MODEL_PATH   = "C:/Users/NITIKA KUMARI/instrunet-ai/models/instrunet_cnn_v3.keras"
DATASET_PATH = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"
CLASS_NAMES  = ['cel','cla','flu','gac','gel','org','pia','sax','tru','vio','voi']
CLASS_FULL   = {
    'cel':'Cello',           'cla':'Clarinet',
    'flu':'Flute',           'gac':'Acoustic Guitar',
    'gel':'Electric Guitar', 'org':'Organ',
    'pia':'Piano',           'sax':'Saxophone',
    'tru':'Trumpet',         'vio':'Violin',
    'voi':'Voice'
}

os.makedirs("outputs", exist_ok=True)


# ── Load val file paths ─────────────────────────────────────────
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

_, val_paths, _, val_labels = train_test_split(
    file_paths, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(f"Validation samples : {len(val_paths)}")


# ── Load model ──────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)


# ── Run predictions ─────────────────────────────────────────────
print("Running predictions...")
y_true_list = []
y_prob_list = []

for i, (path, label) in enumerate(zip(val_paths, val_labels)):
    features      = np.load(path).astype(np.float32)
    mean, var     = np.mean(features), np.var(features)
    features      = (features - mean) / (np.sqrt(var) + 1e-7)
    features      = np.expand_dims(features, axis=0)
    probs         = model.predict(features, verbose=0)[0]
    y_true_list.append(label)
    y_prob_list.append(probs)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(val_paths)} done")

y_true_raw = np.array(y_true_list)
y_prob     = np.array(y_prob_list)

# binarize for ROC
y_true = label_binarize(y_true_raw, classes=list(range(11)))


# ── Plot ROC curves ─────────────────────────────────────────────
colors = ['#4A90D9','#2CA87F','#E09132','#D94A4A','#9B4AD9',
          '#4AD9C8','#D9A84A','#4A6ED9','#D94A9B','#7AD94A','#888780']

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

all_fpr = []
all_tpr = []
all_auc = []

for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
    roc_auc     = auc(fpr, tpr)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

    ax = axes[i]
    ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0,1],[0,1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_title(CLASS_FULL[cls], fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

# macro average
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(mean_fpr)
for i in range(len(CLASS_NAMES)):
    mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
mean_tpr /= len(CLASS_NAMES)
mean_auc  = auc(mean_fpr, mean_tpr)

ax = axes[11]
for i, cls in enumerate(CLASS_NAMES):
    ax.plot(all_fpr[i], all_tpr[i], color=colors[i],
            linewidth=1, alpha=0.4, label=f'{cls} ({all_auc[i]:.2f})')
ax.plot(mean_fpr, mean_tpr, color='black',
        linewidth=2.5, label=f'Macro avg ({mean_auc:.3f})')
ax.plot([0,1],[0,1], 'k--', linewidth=0.8, alpha=0.5)
ax.set_title('Macro Average ROC', fontsize=11, fontweight='bold')
ax.set_xlabel('False Positive Rate', fontsize=9)
ax.set_ylabel('True Positive Rate', fontsize=9)
ax.legend(loc='lower right', fontsize=7)
ax.grid(True, alpha=0.3)

plt.suptitle('ROC Curves — InstruNet Instrument Classifier',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Print AUC scores ────────────────────────────────────────────
print("\n── ROC AUC Scores ──────────────────────────")
for i, cls in enumerate(CLASS_NAMES):
    bar = '█' * int(all_auc[i] * 20)
    print(f"  {CLASS_FULL[cls]:<20} AUC: {all_auc[i]:.3f}  {bar}")
print(f"\n  Macro Average AUC : {mean_auc:.3f}")
print("────────────────────────────────────────────")
print("\nROC curve saved → outputs/roc_curves.png")