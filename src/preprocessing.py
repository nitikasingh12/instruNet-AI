# -------------------------------------------------------
# IMPORT REQUIRED MODULES
# -------------------------------------------------------

# dataset loader
from dataset_builder import build_dataset

# CNN model
from model import build_model

# tensorflow tools
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# plotting
import matplotlib.pyplot as plt

# numerical operations
import numpy as np



# -------------------------------------------------------
# TRAINING HYPERPARAMETERS
# -------------------------------------------------------

EPOCHS = 10                     # train model for 10 epochs
LEARNING_RATE = 0.0003          # lower learning rate improves stability
DATASET_PATH = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms"
MODEL_SAVE_PATH = "models/instrunet_cnn.keras"



# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------

train_data, val_data = build_dataset(DATASET_PATH)

# number of classes
num_classes = train_data.num_classes

# class names
class_names = list(train_data.class_indices.keys())

print("\nDetected Classes:")
print(class_names)



# -------------------------------------------------------
# BUILD CNN MODEL
# -------------------------------------------------------

model = build_model(num_classes)

print("\nModel Architecture:\n")
model.summary()



# -------------------------------------------------------
# COMPILE MODEL
# -------------------------------------------------------

model.compile(

    optimizer=Adam(learning_rate=LEARNING_RATE),   # optimizer updates weights

    loss="categorical_crossentropy",                # multi-class classification loss

    metrics=["accuracy"]                            # track accuracy
)



# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------

# stop training if validation loss stops improving
early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)


# save best model
checkpoint = ModelCheckpoint(

    MODEL_SAVE_PATH,

    monitor="val_accuracy",

    save_best_only=True
)



# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------

print("\nStarting Training...\n")

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint]
)



# -------------------------------------------------------
# PLOT TRAINING vs VALIDATION CURVES
# -------------------------------------------------------

# accuracy plot
plt.figure()

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()



# loss plot
plt.figure()

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()



# -------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------

print("\nEvaluating Model Performance...\n")

# reset validation generator
val_data.reset()

# generate predictions
predictions = model.predict(val_data)

# predicted labels
y_pred = np.argmax(predictions, axis=1)

# true labels
y_true = val_data.classes



# -------------------------------------------------------
# OVERALL ACCURACY
# -------------------------------------------------------

overall_accuracy = accuracy_score(y_true, y_pred)

print("\nOverall Accuracy:")

print(overall_accuracy)



# -------------------------------------------------------
# PER CLASS METRICS
# -------------------------------------------------------

print("\nPer-Instrument Metrics:\n")

report = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    zero_division=0
)

print(report)



# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------

print("\nConfusion Matrix:\n")

cm = confusion_matrix(y_true, y_pred)

print(cm)



# -------------------------------------------------------
# SAVE FINAL MODEL
# -------------------------------------------------------

model.save(MODEL_SAVE_PATH)

print("\nModel saved successfully at:", MODEL_SAVE_PATH)