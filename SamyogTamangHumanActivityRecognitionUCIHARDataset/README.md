**## SamyogTamangHumanActivityRecognitionUCIHARDataset

## Student
Name:Samyog Tamang

## Project Title
Title: Human Activity Recognition

## Objective
Objective To recognize Human Activity using UCIHAR Dataset
## Dataset
Dataset : UCIHAR Dataset

## ANN Architecture
The model is a fully-connected feedforward network built with scikit-learn's MLPClassifier (a TensorFlow-free stand-in for a Keras Sequential ANN):

Input layer: 561 standardized features (time- and frequency-domain signals from the UCI HAR dataset, scaled with StandardScaler)
Hidden layers: 128 → 64 → 32 units, all with ReLU activation
Output layer: 6 units (one per activity class), with softmax applied internally by scikit-learn for multi-class output
Optimizer: Adam
Loss function: log-loss (categorical cross-entropy equivalent)
Regularization: L2 penalty (alpha=1e-4) plus early stopping on a 15% held-out validation split — filling the role Dropout plays in a typical Keras version
Training config: batch size 32, up to 60 iterations (epochs), n_iter_no_change=60 so early stopping doesn't cut training off prematurely

## Results
Test accuracy: 99%, with log-loss also reported at evaluation time
Per-class precision/recall/F1 were all in the 0.97–1.00 range across all six activities
LAYING was classified perfectly (1.00 precision/recall); the small amount of confusion that did occur was concentrated between SITTING and STANDING (0.97–0.99 range), which is the expected failure mode for HAR since those two postures produce very similar sensor signatures
The validation accuracy curve rose and plateaued smoothly over training epochs, and the training loss curve decreased steadily, with no signs of overfitting
A single held-out test sample was run through the trained pipeline (scaler → model) and correctly predicted as WALKING_UPSTAIRS with 100% confidence, demonstrating the full inference path end-to-end
## Conclusion
This project showed that an ANN can classify six human activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) from smartphone accelerometer/gyroscope features with very high accuracy. Standardizing the 561 pre-engineered features and training a 128→64→32 MLP with Adam, L2 regularization, and early stopping was sufficient to reach ~99% test accuracy, with nearly all residual error confined to the SITTING/STANDING pair — a known hard case in HAR due to postural similarity. The resulting pipeline (scaler + model) generalizes to new sensor readings and could be deployed for real-time activity recognition from a live feature-extraction pipeline.
**
