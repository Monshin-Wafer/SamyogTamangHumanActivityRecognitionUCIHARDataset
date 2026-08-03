import pandas as pd

# Load dataset
data = pd.read_csv("../dataset/spam.csv", encoding="latin-1")

# Keep only the first two columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Display information
print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nClass Distribution:")
print(data["label"].value_counts())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("../dataset/spam.csv", encoding="latin-1")

# Keep only first two columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Convert labels into numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split features and labels
X = data["message"]
y = data["label"]

# Convert text into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Convert sparse matrix to dense array
X_train = X_train.toarray()
X_test = X_test.toarray()

# Build ANN
model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("../report/results/training_curves.png")
plt.show()
# Predict
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../report/results/confusion_matrix.png")
plt.show()

# Precision, Recall, F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

print("Accuracy:", accuracy)
import joblib

# Save the trained model
model.save("../model/spam_ann.keras")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "../model/vectorizer.pkl")

print("ANN model and vectorizer saved successfully!")