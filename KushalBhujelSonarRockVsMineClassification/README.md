# KushalBhujelSonarRockVsMineClassification


# Sonar Rock vs Mine Classification using ANN

Kushal Bhujel

## Objective
Build a simple neural network to classify sonar signals as rocks or mines. This is a beginner-friendly project to learn the basics of deep learning.

## Problem Statement
We need to create a model that takes sonar data as input and decides if it's detecting a rock or a mine. The input is 60 numbers representing different aspects of the sonar signal, and the output is either "rock" (0) or "mine" (1). This is actually useful for real naval operations where automatic mine detection can save lives.

## Dataset Description
Using the classic Sonar dataset from UCI Machine Learning Repository:

- 208 samples total
- 60 numerical features (sonar signal energy at different frequencies)
- 1 target column (R for Rock, M for Mine)
- 111 mines, 97 rocks (well-balanced)

Each feature represents energy in a specific frequency band. Values are normalized between 0 and 1.

## Data Preprocessing
The dataset was clean:
- No missing values
- Converted R to 0 and M to 1 using LabelEncoder
- Split into features (X) and target (y)

## Feature Scaling
Used StandardScaler to normalize all features to have mean=0 and std=1. This helps the neural network learn better by putting all features on the same scale.

## Train-Test Split
- 80% training (166 samples)
- 20% testing (42 samples)
- random_state=42 for reproducible results

## ANN Model
Simple 3-layer network:

Input: 60 neurons
↓
Hidden: 32 neurons (ReLU)
↓
Hidden: 16 neurons (ReLU)
↓
Output: 1 neuron (Sigmoid)

- 2,497 total parameters
- Adam optimizer
- Binary crossentropy loss
- Accuracy metric

## Training and Validation Graphs
Trained for 50 epochs:

- Training accuracy: 53% → 100%
- Validation accuracy: 62% → 76.47%
- Training loss: 0.73 → 0.036
- Validation loss: 0.73 → 0.490

Generated accuracy.png and loss.png

## Accuracy/Loss Analysis
Test results:
- Test accuracy: 80.95%
- Test loss: 0.5196

The model performs reasonably well. There's some overfitting (training at 100% vs test at 81%), but that's expected with a small dataset.

## Confusion Matrix
Test set performance:


                Predicted
              Rock  Mine
Actual Rock     19     7
      Mine      1    15


- 19 rocks correctly identified
- 7 rocks called mines (false positives)
- 1 mine called rock (false negative)
- 15 mines correctly identified




## Precision, Recall, F1-score

              precision    recall  f1-score   support
        Rock       0.95      0.73      0.83        26
        Mine       0.68      0.94      0.79        16
    accuracy                           0.81        42


- Rock precision: 95% (when it says rock, it's usually right)
- Mine recall: 94% (catches most actual mines)
- F1-scores around 80% for both classes

Full report in classification_report.txt

## Prediction on New Sample
Tested on one sample:
- Actual: Rock
- Predicted: Rock
- Confidence: 0.0007 (very confident)
- Result: Correct

Details in prediction.txt

## Conclusion
This project shows that a simple neural network can classify sonar signals pretty well. We got 81% accuracy with just 3 layers. The code is clean and easy to understand, making it great for learning.

What worked:
- Simple architecture was effective
- Feature scaling helped training
- Good accuracy for a basic model

Could improve:
- Add dropout to reduce overfitting
- Tune hyperparameters
- Try cross-validation

Real uses:
- Naval mine detection
- Underwater exploration
- Autonomous underwater vehicles

