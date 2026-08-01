"""Load the trained ANN and predict the species of new iris measurements."""

import os
import sys

import numpy as np
import joblib
from tensorflow import keras
from sklearn.datasets import load_iris

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

TARGET_NAMES = load_iris().target_names

# sepal_length, sepal_width, petal_length, petal_width (cm)
SAMPLES = np.array(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.7, 5.1, 1.6],
        [6.7, 3.3, 5.7, 2.5],
    ]
)


def main():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    samples_scaled = scaler.transform(SAMPLES)
    predictions = np.argmax(model.predict(samples_scaled, verbose=0), axis=1)

    for sample, pred in zip(SAMPLES, predictions):
        print(f"Measurements {sample.tolist()} -> Predicted species: {TARGET_NAMES[pred]}")


if __name__ == "__main__":
    main()
