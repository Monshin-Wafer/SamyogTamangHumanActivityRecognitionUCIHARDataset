# HistManandharParkinsonsDiseaseDetection

## Student
**Name:** Hist Manandhar

---

## Project Title

**Parkinson Disease Detection using Artificial Neural Network (ANN)**

---

## Objective

To build an Artificial Neural Network (ANN) model that predicts whether a person has Parkinson's disease based on biomedical voice measurements.

---

## Dataset

- **Dataset Name:** Parkinson's Disease Dataset
- **Source:** UCI Machine Learning Repository
- **File:** `parkinsons.data`
- **Target Variable:** `status`
  - `1` = Parkinson's Disease
  - `0` = Healthy

The dataset contains several biomedical voice measurements such as jitter, shimmer, NHR, HNR, RPDE, DFA, and PPE.

---

## ANN Architecture

- Input Layer
- Hidden Layer 1 (ReLU)
- Hidden Layer 2 (ReLU)
- Output Layer (Sigmoid)

**Optimizer:** Adam

**Loss Function:** Binary Crossentropy

**Evaluation Metric:** Accuracy

---

## Results

The ANN model was trained and tested using the Parkinson's disease dataset.

The project includes:
- Data preprocessing
- Feature scaling
- Train-test split
- ANN model training
- Model evaluation
- Accuracy and loss visualization
- Prediction of Parkinson's disease

---

## Conclusion

This project demonstrates the use of an Artificial Neural Network (ANN) for Parkinson's disease detection using biomedical voice features. The trained model can classify patients as healthy or affected by Parkinson's disease and demonstrates the effectiveness of deep learning in medical diagnosis.