"""
Sonar Rock vs Mine Classification 

"""

# ============================================================
# 1. Import Libraries
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories if they don't exist
os.makedirs('../model', exist_ok=True)
os.makedirs('../results', exist_ok=True)

print("Libraries imported successfully!")
print("=" * 60)


# ============================================================
# 2. Load Dataset
# ============================================================
print("\n2. Loading Dataset...")

# Load the sonar dataset from CSV file
# The dataset has 60 features and 1 target column (R for Rock, M for Mine)
df = pd.read_csv('../dataset/sonar.csv', header=None)

print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"First few rows:")
print(df.head())

print("=" * 60)


# ============================================================
# 3. Dataset Description
# ============================================================
print("\n3. Dataset Description...")

# Rename columns for better understanding (feature_0 to feature_59, and target)
feature_columns = [f'feature_{i}' for i in range(60)]
df.columns = feature_columns + ['target']

# Display basic information about the dataset
print(f"Total number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"Target column: {df.columns[-1]}")

# Show target distribution
print(f"\nTarget distribution:")
print(df['target'].value_counts())

# Display statistical summary
print(f"\nStatistical summary of features:")
print(df.describe())

print("=" * 60)


# ============================================================
# 4. Data Preprocessing
# ============================================================
print("\n4. Data Preprocessing...")

# Check for missing values
print(f"Missing values in each column:")
print(df.isnull().sum())

# Since there are no missing values, we proceed with label encoding
# Encode the target variable: R (Rock) = 0, M (Mine) = 1
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

print(f"Target after encoding (0=Rock, 1=Mine):")
print(df['target'].value_counts())

# Separate features (X) and target (y)
X = df.drop('target', axis=1)  # All columns except target
y = df['target']  # Only the target column

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("=" * 60)


# ============================================================
# 5. Feature Scaling
# ============================================================
print("\n5. Feature Scaling...")

# Use StandardScaler to normalize features (mean=0, std=1)
# This helps the neural network converge faster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature scaling completed using StandardScaler")
print(f"Scaled features shape: {X_scaled.shape}")
print(f"Sample of scaled features (first row):")
print(X_scaled[0])

print("=" * 60)


# ============================================================
# 6. Train-Test Split
# ============================================================
print("\n6. Train-Test Split...")

# Split the dataset into 80% training and 20% testing
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training target distribution:")
print(y_train.value_counts())
print(f"Testing target distribution:")
print(y_test.value_counts())

print("=" * 60)


# ============================================================
# 7. Build ANN Model
# ============================================================
print("\n7. Building ANN Model...")

# Create a Sequential model (layers are added one after another)
model = Sequential()

# Input layer
# Input shape is 60 (number of features)
model.add(Input(shape=(60,)))

# First hidden layer
# Dense layer with 32 neurons and ReLU activation function
model.add(Dense(32, activation='relu'))

# Second hidden layer
# Dense layer with 16 neurons and ReLU activation function
model.add(Dense(16, activation='relu'))

# Output layer
# Single neuron with sigmoid activation for binary classification
# Sigmoid outputs a value between 0 and 1 (probability)
model.add(Dense(1, activation='sigmoid'))

print("ANN Model Architecture:")
model.summary()

print("=" * 60)


# ============================================================
# 8. Compile Model
# ============================================================
print("\n8. Compiling Model...")

# Compile the model with optimizer, loss function, and metrics
# Adam optimizer: adaptive learning rate optimization algorithm
# binary_crossentropy: loss function for binary classification
# accuracy: metric to monitor during training
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")
print("Optimizer: Adam")
print("Loss function: binary_crossentropy")
print("Metric: accuracy")

print("=" * 60)


# ============================================================
# 9. Train Model
# ============================================================
print("\n9. Training Model...")

# Train the model for 50 epochs with 20% validation split
# epochs: number of times the model sees the entire training data
# validation_split: portion of training data used for validation
# batch_size: default is 32 (number of samples per gradient update)
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

print("Model training completed!")

print("=" * 60)


# ============================================================
# 10. Training & Validation Graphs
# ============================================================
print("\n10. Generating Training & Validation Graphs...")

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save accuracy plot
plt.savefig('../results/accuracy.png', dpi=300, bbox_inches='tight')
print("Accuracy graph saved to ../results/accuracy.png")

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save loss plot
plt.savefig('../results/loss.png', dpi=300, bbox_inches='tight')
print("Loss graph saved to ../results/loss.png")

plt.tight_layout()
# Comment out plt.show() to prevent blocking in non-interactive environments
# plt.show()

print("=" * 60)


# ============================================================
# 11. Model Evaluation
# ============================================================
print("\n11. Evaluating Model on Test Data...")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Also display final training metrics
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\nFinal Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

print("=" * 60)


# ============================================================
# 12. Confusion Matrix
# ============================================================
print("\n12. Generating Confusion Matrix...")

# Make predictions on test data
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rock (0)', 'Mine (1)'],
            yticklabels=['Rock (0)', 'Mine (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to ../results/confusion_matrix.png")
# Comment out plt.show() to prevent blocking in non-interactive environments
# plt.show()

print("Confusion Matrix Values:")
print(cm)

print("=" * 60)


# ============================================================
# 13. Precision, Recall, F1-score
# ============================================================
print("\n13. Calculating Precision, Recall, F1-score...")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Rock', 'Mine'], output_dict=True)
report_text = classification_report(y_test, y_pred, target_names=['Rock', 'Mine'])

# Print the classification report
print("Classification Report:")
print(report_text)

# Save classification report to file
with open('../results/classification_report.txt', 'w') as f:
    f.write("Classification Report for Sonar Rock vs Mine Classification\n")
    f.write("=" * 60 + "\n\n")
    f.write(report_text)

print("Classification report saved to ../results/classification_report.txt")

print("=" * 60)


# ============================================================
# 14. Prediction on New Sample
# ============================================================
print("\n14. Prediction on New Sample...")

# Take one sample from the test set for prediction
sample_index = 0
new_sample = X_test[sample_index]
actual_label = y_test.iloc[sample_index] if hasattr(y_test, 'iloc') else y_test[sample_index]

# Scale the sample (already scaled since it's from X_test)
# Reshape for prediction (model expects 2D array)
sample_reshaped = new_sample.reshape(1, -1)

# Make prediction
prediction_prob = model.predict(sample_reshaped, verbose=0)[0][0]
prediction_label = 1 if prediction_prob > 0.5 else 0

# Convert to human-readable labels
actual_name = "Mine" if actual_label == 1 else "Rock"
predicted_name = "Mine" if prediction_label == 1 else "Rock"

print(f"Sample Index: {sample_index}")
print(f"Actual Label: {actual_name} ({actual_label})")
print(f"Predicted Label: {predicted_name} ({prediction_label})")
print(f"Prediction Confidence: {prediction_prob:.4f}")
print(f"Prediction Result: {'CORRECT' if actual_label == prediction_label else 'INCORRECT'}")

# Save prediction result to file
with open('../results/prediction.txt', 'w') as f:
    f.write("Prediction on New Sample\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Sample Index: {sample_index}\n")
    f.write(f"Actual Label: {actual_name} ({actual_label})\n")
    f.write(f"Predicted Label: {predicted_name} ({prediction_label})\n")
    f.write(f"Prediction Confidence: {prediction_prob:.4f}\n")
    result_status = "CORRECT" if actual_label == prediction_label else "INCORRECT"
    f.write(f"Prediction Result: {result_status}\n")

print("Prediction result saved to ../results/prediction.txt")

print("=" * 60)


# ============================================================
# 15. Save Model
# ============================================================
print("\n15. Saving Trained Model...")

# Save the trained model to disk
model.save('../model/sonar_ann.keras')

print("Model saved successfully to ../model/sonar_ann.keras")

print("=" * 60)


# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nSummary:")
print(f"- Model Architecture: 3-layer ANN (60-32-16-1)")
print(f"- Training Epochs: 50")
print(f"- Test Accuracy: {test_accuracy:.4f}")
print(f"- Test Loss: {test_loss:.4f}")
print("\nGenerated Files:")
print("- Model: ../model/sonar_ann.keras")
print("- Results: ../results/accuracy.png")
print("- Results: ../results/loss.png")
print("- Results: ../results/confusion_matrix.png")
print("- Results: ../results/classification_report.txt")
print("- Results: ../results/prediction.txt")
print("\nThank you for using this ANN classification project!")
print("=" * 60)
