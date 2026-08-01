"""
Iris Flower Classification using an Artificial Neural Network (TensorFlow/Keras)
==================================================================================
Trains an ANN to classify iris flowers into setosa, versicolor, or virginica
from four measurements, and saves the model, scaler, and evaluation plots.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "report", "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
sns.set_style("whitegrid")


# --------------------------------------------------------------------------- #
# 1. Dataset
# --------------------------------------------------------------------------- #
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("Shape:", df.shape)
print("Class distribution:\n", df["species"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="species", hue="species", data=df, palette="viridis", legend=False)
plt.title("Class Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=150)
plt.close()


# --------------------------------------------------------------------------- #
# 2. Data preprocessing
# --------------------------------------------------------------------------- #
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

X = df[iris.feature_names].values
y = df["target"].values


# --------------------------------------------------------------------------- #
# 3. Train-test split
# --------------------------------------------------------------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)


# --------------------------------------------------------------------------- #
# 4. Feature scaling (fit on train only, to avoid leakage)
# --------------------------------------------------------------------------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))


# --------------------------------------------------------------------------- #
# 5. ANN model
# --------------------------------------------------------------------------- #
model = keras.Sequential(
    [
        layers.Input(shape=(4,), name="input_features"),
        layers.Dense(16, activation="relu", name="hidden_1"),
        layers.Dense(8, activation="relu", name="hidden_2"),
        layers.Dense(3, activation="softmax", name="output"),
    ],
    name="iris_ann",
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# --------------------------------------------------------------------------- #
# 6. Training
# --------------------------------------------------------------------------- #
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=8,
    callbacks=[early_stop],
    verbose=0,
)
print(f"Training stopped after {len(history.history['loss'])} epochs.")


# --------------------------------------------------------------------------- #
# 7. Training / validation graphs
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
plt.close()


# --------------------------------------------------------------------------- #
# 8. Accuracy / loss analysis
# --------------------------------------------------------------------------- #
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")


# --------------------------------------------------------------------------- #
# 9. Confusion matrix
# --------------------------------------------------------------------------- #
y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names, yticklabels=iris.target_names,
)
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()


# --------------------------------------------------------------------------- #
# 10. Precision, recall, F1-score
# --------------------------------------------------------------------------- #
report_text = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report_text)
with open(os.path.join(BASE_DIR, "report", "classification_report.txt"), "w") as f:
    f.write(report_text)


# --------------------------------------------------------------------------- #
# 11. Save model + run summary
# --------------------------------------------------------------------------- #
model.save(os.path.join(MODEL_DIR, "iris_ann.keras"))

summary = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "epochs_run": len(history.history["loss"]),
}
with open(os.path.join(BASE_DIR, "report", "run_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Model, scaler, and plots saved.")
