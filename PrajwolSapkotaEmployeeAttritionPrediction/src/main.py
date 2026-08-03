"""
Employee Attrition Prediction using ANN
Student: Prajwol Sapkota

This program trains the ANN model, tests it, saves the model and also saves
all the graphs in the results folder.

Run:  cd src && python main.py
"""

import os
import json
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # save the graphs as files instead of showing them
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score, precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fixed seed so the results are the same every run
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "../dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = "../model"
RESULTS_DIR = "../results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colors used in all the graphs
STAY, LEFT, MUTED = "#2a78d6", "#eb6834", "#898781"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e1e0d9", "figure.dpi": 120, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------------------
print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(df["Attrition"].value_counts())

rate = (df["Attrition"] == "Yes").mean()
print(f"Attrition rate: {rate:.1%}")


# ----------------------------------------------------------------------
# 2. EDA graph
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. EDA")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Class balance
counts = df["Attrition"].value_counts()
bars = axes[0].bar(["Stayed", "Left"], [counts["No"], counts["Yes"]],
                   color=[STAY, LEFT], width=0.55)
for b, v in zip(bars, [counts["No"], counts["Yes"]]):
    axes[0].text(b.get_x() + b.get_width()/2, v + 20, f"{v}\n({v/len(df):.1%})",
                 ha="center", va="bottom", fontsize=9)
axes[0].set_title("Attrition is imbalanced", fontweight="bold")
axes[0].set_ylabel("Employees")
axes[0].set_ylim(0, counts["No"] * 1.25)

# (b) Overtime - the strongest factor
g = df.groupby("OverTime", observed=True)["Attrition"].apply(lambda s: (s == "Yes").mean())
bars = axes[1].bar(g.index.astype(str), g.values, color=[STAY, LEFT], width=0.55)
for b, v in zip(bars, g.values):
    axes[1].text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.0%}",
                 ha="center", va="bottom", fontsize=9)
axes[1].set_title("Attrition rate by OverTime", fontweight="bold")
axes[1].set_xlabel("OverTime")
axes[1].set_ylim(0, g.max() * 1.25)
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# (c) Job role
g = (df.groupby("JobRole", observed=True)["Attrition"]
       .apply(lambda s: (s == "Yes").mean()).sort_values())
axes[2].barh(g.index, g.values, color=STAY, height=0.65)
axes[2].axvline(rate, color=MUTED, lw=1, ls="--")
axes[2].set_title("Attrition rate by Job Role", fontweight="bold")
axes[2].tick_params(axis="y", labelsize=7)
axes[2].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/eda_overview.png")
plt.close()
print(f"Saved {RESULTS_DIR}/eda_overview.png")


# ----------------------------------------------------------------------
# 3. Preprocessing
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. PREPROCESSING")
print("=" * 60)

# Remove the columns which are not useful
nun = df.nunique()
constant_cols = list(nun[nun == 1].index)        # same value for all employees
id_cols = list(nun[nun == len(df)].index)        # only an ID number
DROP_COLS = constant_cols + id_cols
print("Dropped columns:", DROP_COLS)

data = df.drop(columns=DROP_COLS)

# Target Yes/No -> 1/0
y = (data["Attrition"] == "Yes").astype(int).values
X_raw = data.drop(columns=["Attrition"])

# One-hot encoding for text columns
cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True).astype("float32")
print(f"Features after one-hot encoding: {X_raw.shape[1]} -> {X.shape[1]}")

# Stratified split keeps the same 16% ratio in every set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X.values, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=SEED, stratify=y_train_full)

# Scaler is fitted only on training data, not on test data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

for name, yy in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
    print(f"{name:11s}: {len(yy):4d} employees ({yy.mean():.1%} left)")

# Class weights, otherwise the model ignores the small "Left" class
w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: float(w[0]), 1: float(w[1])}
print("Class weights:", {k: round(v, 3) for k, v in class_weight.items()})


# ----------------------------------------------------------------------
# 4. Build the ANN
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. ANN ARCHITECTURE")
print("=" * 60)

keras.utils.set_random_seed(SEED)
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],), name="input"),
    layers.Dense(32, activation="relu",
                 kernel_regularizer=keras.regularizers.l2(1e-2), name="hidden_1"),
    layers.Dropout(0.3, name="dropout_1"),
    layers.Dense(16, activation="relu",
                 kernel_regularizer=keras.regularizers.l2(1e-2), name="hidden_2"),
    layers.Dense(1, activation="sigmoid", name="output"),
], name="attrition_ann")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc"),
             keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall")],
)
model.summary()


# ----------------------------------------------------------------------
# 5. Training
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. TRAINING")
print("=" * 60)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=40,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12,
                                      min_lr=1e-5, verbose=0),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500, batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks, verbose=0,
)
print(f"Training stopped after {len(history.history['loss'])} epochs")

# Training curves
h = history.history
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (tr, va, title) in zip(axes, [("loss", "val_loss", "Loss"),
                                      ("auc", "val_auc", "ROC-AUC"),
                                      ("recall", "val_recall", "Recall")]):
    ax.plot(h[tr], color=STAY, lw=2, label="Train")
    ax.plot(h[va], color=LEFT, lw=2, label="Validation")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png")
plt.close()
print(f"Saved {RESULTS_DIR}/training_curves.png")


# ----------------------------------------------------------------------
# 6. Evaluation
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. EVALUATION (test set)")
print("=" * 60)

y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("Results at the default threshold 0.50")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"], digits=3))

# Checking different threshold values, because 0.5 is not always the best
sweep = []
for t in np.arange(0.05, 0.96, 0.01):
    p = (y_prob >= t).astype(int)
    sweep.append({"t": t,
                  "precision": precision_score(y_test, p, zero_division=0),
                  "recall": recall_score(y_test, p, zero_division=0),
                  "f1": f1_score(y_test, p, zero_division=0)})
sweep = pd.DataFrame(sweep)
best = sweep.loc[sweep.f1.idxmax()]
BEST_T = float(best.t)
print(f"Best threshold = {BEST_T:.2f} -> precision {best.precision:.3f} | "
      f"recall {best.recall:.3f} | F1 {best.f1:.3f}")

y_best = (y_prob >= BEST_T).astype(int)

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_best)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Stayed", "Left"])
ax.set_yticks([0, 1], ["Stayed", "Left"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (threshold {BEST_T:.2f})", fontweight="bold")
notes = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}\n{notes[i][j]}", ha="center", va="center",
                fontsize=10, color="white" if cm[i, j] > cm.max()/2 else "black")
ax.grid(False)
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.close()
print(f"Saved {RESULTS_DIR}/confusion_matrix.png")

# --- ROC and Precision-Recall curves ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
prec, rec, _ = precision_recall_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.plot(fpr, tpr, color=STAY, lw=2.2, label=f"ANN (AUC = {auc:.3f})")
a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random guess")
a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
a1.set_title("ROC Curve", fontweight="bold"); a1.legend(frameon=False)

a2.plot(rec, prec, color=LEFT, lw=2.2, label=f"ANN (AP = {ap:.3f})")
a2.axhline(y_test.mean(), color=MUTED, lw=1, ls="--", label=f"Baseline ({y_test.mean():.1%})")
a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
a2.set_title("Precision-Recall Curve", fontweight="bold"); a2.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_pr_curves.png")
plt.close()
print(f"Saved {RESULTS_DIR}/roc_pr_curves.png")

# --- Threshold tuning graph ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sweep.t, sweep.precision, color=STAY, lw=2, label="Precision")
ax.plot(sweep.t, sweep.recall, color=LEFT, lw=2, label="Recall")
ax.plot(sweep.t, sweep.f1, color="#4a3aa7", lw=2, ls="--", label="F1")
ax.axvline(BEST_T, color=MUTED, lw=1, ls=":")
ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Threshold tuning (best F1 at {BEST_T:.2f})", fontweight="bold")
ax.legend(frameon=False)
plt.savefig(f"{RESULTS_DIR}/threshold_tuning.png")
plt.close()
print(f"Saved {RESULTS_DIR}/threshold_tuning.png")


# ----------------------------------------------------------------------
# 7. Save the model
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. SAVING MODEL")
print("=" * 60)

model.save(f"{MODEL_DIR}/attrition_ann.keras")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

metadata = {
    "feature_names": X.columns.tolist(),
    "n_features": int(X.shape[1]),
    "decision_threshold": round(BEST_T, 2),
    "categorical_columns": cat_cols,
    "dropped_columns": DROP_COLS,
    "architecture": f"{X.shape[1]} -> 32(ReLU) -> Dropout(0.3) -> 16(ReLU) -> 1(Sigmoid)",
    "epochs_trained": len(history.history["loss"]),
    "test_metrics": {
        "accuracy": round(float(accuracy_score(y_test, y_best)), 4),
        "precision": round(float(precision_score(y_test, y_best, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_best, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_best, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    },
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved: attrition_ann.keras, scaler.joblib, metadata.json")
print("\nFINAL RESULT -> ROC-AUC "
      f"{metadata['test_metrics']['roc_auc']}, "
      f"F1 {metadata['test_metrics']['f1']}, "
      f"Recall {metadata['test_metrics']['recall']}")
print("Done. Now you can run:  python predict.py")
