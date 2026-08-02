"""Load the trained ANN and predict Pass/Fail for a new student record."""

import os

import pandas as pd
import joblib
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_performance_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

BINARY_MAPS = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"GT3": 0, "LE3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "schoolsup": {"no": 0, "yes": 1},
    "famsup": {"no": 0, "yes": 1},
    "paid": {"no": 0, "yes": 1},
    "activities": {"no": 0, "yes": 1},
    "nursery": {"no": 0, "yes": 1},
    "higher": {"no": 0, "yes": 1},
    "internet": {"no": 0, "yes": 1},
    "romantic": {"no": 0, "yes": 1},
}
MULTI_CATEGORY_COLS = ["Mjob", "Fjob", "reason", "guardian"]

# A new (unseen) student record, in the same raw format as the dataset columns.
NEW_STUDENT = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 4, "Fedu": 3, "Mjob": "health", "Fjob": "other",
    "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 3,
    "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1,
    "Walc": 1, "health": 5, "absences": 2, "G1": 14, "G2": 15,
}


def preprocess(sample: dict, feature_cols: list) -> pd.DataFrame:
    row = dict(sample)
    for col, mapping in BINARY_MAPS.items():
        row[col] = mapping[row[col]]

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=MULTI_CATEGORY_COLS, drop_first=False)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df.astype(float)


def main():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    X = preprocess(NEW_STUDENT, feature_cols)
    X_scaled = scaler.transform(X.values)
    prob = float(model.predict(X_scaled, verbose=0)[0][0])
    label = "Pass" if prob >= 0.5 else "Fail"

    print(f"Predicted outcome: {label} (pass probability: {prob:.4f})")


if __name__ == "__main__":
    main()
