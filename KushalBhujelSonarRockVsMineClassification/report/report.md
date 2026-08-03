# Sonar Rock vs Mine Classification - Project Report

## Objective
The goal of this project is to build a simple neural network that can tell the difference between rocks and mines using sonar data. It's meant to be a beginner-friendly introduction to deep learning.

## Problem Statement
We need to create a model that looks at sonar signals and decides if they're coming from a rock or a mine. The model gets 60 numbers as input (representing different aspects of the sonar signal) and outputs either "rock" (0) or "mine" (1). This is actually important for real naval operations - being able to automatically detect mines can save lives and equipment.

## Dataset Description
We're using the classic Sonar dataset from UCI Machine Learning Repository. Here's what it looks like:

- 208 total samples
- 60 features (all numbers representing sonar signal energy at different frequencies)
- 1 target column (R for Rock, M for Mine)
- 111 mines and 97 rocks (pretty balanced)

Each feature represents energy in a specific frequency band, collected at different time periods. The values are already normalized between 0 and 1.

## Data Preprocessing
The dataset was pretty clean already:

- No missing values to deal with
- Used LabelEncoder to convert R to 0 and M to 1
- Split the data into features (X) and target (y)
- All features are continuous numbers

## Feature Scaling
I used StandardScaler to normalize the features. This is important for neural networks because it makes all features have the same scale (mean of 0, standard deviation of 1). Without this, features with larger numbers would dominate the learning process. All 60 features are now on the same scale.

## Train-Test Split
Split the data 80-20:
- 166 samples for training (85 rocks, 81 mines)
- 42 samples for testing (26 rocks, 16 mines)
- Used random_state=42 so results are reproducible

## ANN Model (TensorFlow/Keras)
The model is pretty simple - just 3 layers:

```
Input: 60 neurons (one for each feature)
↓
Hidden Layer 1: 32 neurons with ReLU
↓
Hidden Layer 2: 16 neurons with ReLU
↓
Output: 1 neuron with Sigmoid
```

Total parameters: 2,497 (about 10KB in size)
- Optimizer: Adam
- Loss: Binary Crossentropy (standard for binary classification)
- Metric: Accuracy

## Training and Validation Graphs
Trained for 50 epochs with 20% of training data used for validation:

- Training accuracy went from ~53% to 100%
- Validation accuracy went from ~62% to 76.47%
- Training loss dropped from ~0.73 to 0.036
- Validation loss dropped from ~0.73 to 0.490

Two graphs were generated:
- accuracy.png - shows how accuracy improved over time
- loss.png - shows how loss decreased over time

## Accuracy/Loss Analysis
Test results:
- Test accuracy: 80.95%
- Test loss: 0.5196
- Final training accuracy: 100.0%
- Final validation accuracy: 76.47%

The model performs reasonably well. There's some overfitting (training accuracy is 100% but test is only 81%), but that's pretty common with small datasets like this one.

## Confusion Matrix
Here's how the model did on the test set:

```
                Predicted
              Rock  Mine
Actual Rock     19     7
      Mine      1    15
```

Breaking it down:
- 19 rocks correctly identified as rocks
- 7 rocks incorrectly called mines
- 1 mine incorrectly called a rock
- 15 mines correctly identified as mines

The confusion matrix is saved as confusion_matrix.png

## Precision, Recall, F1-score
Classification results:

```
              precision    recall  f1-score   support
        Rock       0.95      0.73      0.83        26
        Mine       0.68      0.94      0.79        16
    accuracy                           0.81        42
```

What this means:
- Precision: When the model says "mine", it's right 68% of the time. When it says "rock", it's right 95% of the time.
- Recall: The model catches 94% of actual mines but only 73% of actual rocks.
- F1-score: A balance between precision and recall. Both classes are around 80%.

Full report saved as classification_report.txt

## Prediction on New Sample
I tested the model on one sample from the test set:

- Sample index: 0
- Actual: Rock
- Predicted: Rock
- Confidence: 0.0007 (very confident it's a rock)
- Result: Correct

This prediction is saved in prediction.txt

## Conclusion
This project shows that even a simple neural network can do a decent job classifying sonar signals. Here's what worked well:

- The 3-layer architecture was simple but effective
- Feature scaling definitely helped with training
- Got 81% accuracy, which is pretty good for this dataset
- All the code is clean and easy to understand

The model could be improved with:
- Adding dropout to reduce overfitting
- Tuning hyperparameters like learning rate
- Using cross-validation for better evaluation
- Maybe trying different architectures

Real-world applications:
- Naval mine detection
- Underwater exploration
- Autonomous underwater vehicles
- Marine safety systems

## Source Code
The main code is in src/main.py. Here's the basic structure:

python
# Import libraries (numpy, pandas, tensorflow, sklearn, etc.)

# Load the dataset from CSV
df = pd.read_csv('../dataset/sonar.csv', header=None)

# Do some basic data exploration
# Check for missing values
# Encode the target (R=0, M=1)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Input(shape=(60,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile with Adam optimizer and binary crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train for 50 epochs
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Generate graphs and evaluate
# Create confusion matrix
# Calculate precision, recall, F1
# Make a sample prediction

# Save the trained model
model.save('../model/sonar_ann.keras')
```
