"""
====================================================================
DRY BEAN CLASSIFICATION USING ARTIFICIAL NEURAL NETWORK (ANN)
====================================================================
Author : <Your Name>
Course : <Course Name>
Dataset: Dry Bean Dataset (Koklu & Ozkan, 2020), UCI Machine Learning
         Repository. https://doi.org/10.24432/C50S4B

NOTE ON DATA: This run uses Dry_Bean_Dataset_synthetic.csv, a
statistically-realistic stand-in generated to match the published
class sizes and feature relationships (see generate_dataset.py),
because this environment has no network access to download the
official file. To reproduce with the REAL dataset:
    1. Download Dry_Bean_Dataset.xlsx from UCI/Kaggle.
    2. Replace the single read line below with:
           df = pd.read_excel("Dry_Bean_Dataset.xlsx")
   No other code needs to change.
====================================================================
"""

# ----------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_score, recall_score, f1_score,
                              accuracy_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

OUT = "outputs/"
import os
os.makedirs(OUT, exist_ok=True)

# ====================================================================
# 2. PROBLEM STATEMENT (printed for the report / console log)
# ====================================================================
print("=" * 70)
print("PROBLEM STATEMENT")
print("=" * 70)
print("""
Dry beans are graded and priced according to variety, and manual sorting
of visually similar bean types is slow, subjective, and error-prone.
This project builds an Artificial Neural Network (ANN) that classifies
dry bean grains into one of SEVEN varieties (Seker, Barbunya, Bombay,
Cali, Horoz, Sira, Dermason) using 16 geometric and shape features
extracted from bean images by a computer vision system. The goal is to
automate variety identification with high accuracy, replacing manual
inspection with a fast, consistent, data-driven classifier.
""")

# ====================================================================
# 3. DATASET DESCRIPTION
# ====================================================================
print("=" * 70)
print("DATASET DESCRIPTION")
print("=" * 70)

df = pd.read_excel("Dry_Bean_Dataset.xlsx")

print(f"""
Source      : Dry Bean Dataset (Koklu & Ozkan, 2020), UCI ML Repository
Instances   : {df.shape[0]}
Features    : {df.shape[1] - 1} numeric input features + 1 target ('Class')
Classes (7) : {sorted(df['Class'].unique())}

Feature list:
  Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
  Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity,
  roundness, Compactness, ShapeFactor1, ShapeFactor2,
  ShapeFactor3, ShapeFactor4

These are geometric/shape descriptors extracted via image processing
(area in pixels, perimeter, axis lengths, roundness, solidity, etc.)
""")

print("First 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df["Class"].value_counts())
print("\nSummary statistics:")
print(df.describe().T[["mean", "std", "min", "max"]])
print("\nMissing values per column:")
print(df.isnull().sum().sum(), "total missing values")

# Class distribution plot
plt.figure(figsize=(8, 5))
order = df["Class"].value_counts().index
sns.countplot(data=df, x="Class", order=order, hue="Class", palette="viridis", legend=False)
plt.title("Class Distribution - Dry Bean Dataset")
plt.xlabel("Bean Variety")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT + "01_class_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT + "02_correlation_heatmap.png")
plt.close()

# ====================================================================
# 4. DATA PREPROCESSING
# ====================================================================
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# 4.1 Handle duplicates
n_dup = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicate rows removed: {n_dup}")

# 4.2 Handle missing values (none expected, but defensive coding)
df = df.dropna()
print(f"Rows after dropping missing values: {df.shape[0]}")

# 4.3 Separate features (X) and target (y)
X = df.drop(columns=["Class"]).values
y_raw = df["Class"].values

# 4.4 Encode target labels to integers, then one-hot for the ANN
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_
num_classes = len(class_names)
print(f"Classes encoded: {dict(zip(class_names, range(num_classes)))}")

y_onehot = keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# ====================================================================
# 5. TRAIN-TEST SPLIT
# ====================================================================
print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X, y_onehot, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Feature count    : {X_train.shape[1]}")

# ====================================================================
# 6. FEATURE SCALING
# ====================================================================
print("\n" + "=" * 70)
print("FEATURE SCALING (StandardScaler)")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled = scaler.transform(X_test)          # transform test with train stats

print("Feature scaling complete. Example - Area feature before/after:")
print(f"  Before: mean={X_train[:,0].mean():.2f}, std={X_train[:,0].std():.2f}")
print(f"  After : mean={X_train_scaled[:,0].mean():.2f}, std={X_train_scaled[:,0].std():.2f}")

# ====================================================================
# 7. ANN MODEL (TensorFlow / Keras)
# ====================================================================
print("\n" + "=" * 70)
print("BUILDING THE ANN MODEL")
print("=" * 70)

n_features = X_train_scaled.shape[1]

model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(16, activation="relu"),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ====================================================================
# 8. TRAINING
# ====================================================================
print("\n" + "=" * 70)
print("TRAINING THE MODEL")
print("=" * 70)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# ====================================================================
# 9. TRAINING AND VALIDATION GRAPHS
# ====================================================================
print("\n" + "=" * 70)
print("PLOTTING TRAINING / VALIDATION CURVES")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("Model Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("Model Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT + "03_training_validation_curves.png")
plt.close()

# ====================================================================
# 10. ACCURACY / LOSS ANALYSIS ON TEST SET
# ====================================================================
print("\n" + "=" * 70)
print("ACCURACY / LOSS ANALYSIS (TEST SET)")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"Best epoch (lowest val_loss): {best_epoch} / {len(history.history['loss'])}")
print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final val accuracy  : {history.history['val_accuracy'][-1]:.4f}")

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
gap = train_acc - val_acc
if gap > 0.07:
    print(f"NOTE: train-val accuracy gap = {gap:.3f} -> possible overfitting.")
else:
    print(f"NOTE: train-val accuracy gap = {gap:.3f} -> model generalizes well.")

# ====================================================================
# 11. PREDICTIONS + CONFUSION MATRIX
# ====================================================================
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test_int

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Dry Bean ANN Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(OUT + "04_confusion_matrix.png")
plt.close()

print(cm)

# ====================================================================
# 12. PRECISION, RECALL, F1-SCORE
# ====================================================================
print("\n" + "=" * 70)
print("PRECISION, RECALL, F1-SCORE")
print("=" * 70)

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

precision_macro = precision_score(y_true, y_pred, average="macro")
recall_macro = recall_score(y_true, y_pred, average="macro")
f1_macro = f1_score(y_true, y_pred, average="macro")
precision_weighted = precision_score(y_true, y_pred, average="weighted")
recall_weighted = recall_score(y_true, y_pred, average="weighted")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
overall_acc = accuracy_score(y_true, y_pred)

print(f"Overall Accuracy         : {overall_acc:.4f}")
print(f"Macro    Precision/Recall/F1 : {precision_macro:.4f} / {recall_macro:.4f} / {f1_macro:.4f}")
print(f"Weighted Precision/Recall/F1 : {precision_weighted:.4f} / {recall_weighted:.4f} / {f1_weighted:.4f}")

# ====================================================================
# 13. PREDICTION ON A NEW SAMPLE
# ====================================================================
print("\n" + "=" * 70)
print("PREDICTION ON A NEW (UNSEEN) SAMPLE")
print("=" * 70)

# Take one raw example row from the test set (before scaling) to simulate
# a brand-new bean measurement coming from the vision system.
new_sample_raw = X_test[0].reshape(1, -1)
true_label = class_names[y_test_int[0]]

new_sample_scaled = scaler.transform(new_sample_raw)
pred_probs = model.predict(new_sample_scaled, verbose=0)[0]
pred_label = class_names[np.argmax(pred_probs)]

print("Raw feature values of the new sample:")
feature_names = df.drop(columns=["Class"]).columns
for fname, val in zip(feature_names, new_sample_raw[0]):
    print(f"  {fname:18s}: {val:.4f}")

print(f"\nTrue variety      : {true_label}")
print(f"Predicted variety : {pred_label}")
print("\nClass probabilities:")
for cname, prob in sorted(zip(class_names, pred_probs), key=lambda x: -x[1]):
    print(f"  {cname:10s}: {prob*100:6.2f}%")

# ====================================================================
# 14. CONCLUSION
# ====================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
An Artificial Neural Network with two hidden dense layers (64 -> 32 -> 16
neurons, ReLU activations, batch normalization and dropout for
regularization) was trained to classify dry bean grains into 7 varieties
using 16 geometric/shape features. After feature scaling with
StandardScaler and an 80/20 train-test split, the model achieved a test
accuracy of {test_accuracy*100:.2f}% and a weighted F1-score of {f1_weighted:.4f}.
The confusion matrix shows most misclassifications occur between
visually/geometrically similar varieties (e.g. Sira/Dermason,
Barbunya/Cali), which mirrors the difficulty reported in the original
Koklu & Ozkan (2020) study. Overall, the ANN provides a fast and reliable
automated alternative to manual bean-variety sorting, and could be
deployed alongside a computer-vision feature extractor for real-time
grading in agricultural processing pipelines.

Possible future improvements: hyperparameter tuning (grid/Bayesian
search), trying deeper architectures or 1D-CNNs, handling the class
imbalance (Bombay is much smaller than Dermason) with class weights or
SMOTE, and cross-validation for a more robust accuracy estimate.
""")

# Save the trained model
model.save(OUT + "dry_bean_ann_model.keras")
print(f"\nModel saved to {OUT}dry_bean_ann_model.keras")
print("All plots saved to the 'outputs/' folder.")
print("\nDONE.")
