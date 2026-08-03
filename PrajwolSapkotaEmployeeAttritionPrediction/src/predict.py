"""
Predict attrition for one employee using the trained ANN model.
Student: Prajwol Sapkota

Run:  cd src && python predict.py
"""

import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

MODEL_DIR = "../model"
DATA_PATH = "../dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Load the trained model, the scaler and the metadata
model = load_model(f"{MODEL_DIR}/attrition_ann.keras")
scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
with open(f"{MODEL_DIR}/metadata.json") as f:
    meta = json.load(f)

FEATURES = meta["feature_names"]
CAT_COLS = meta["categorical_columns"]
THRESHOLD = meta["decision_threshold"]

# The user only enters a few details, so the remaining columns are filled
# using the median value for numbers and the most common value for text.
df = pd.read_csv(DATA_PATH).drop(columns=meta["dropped_columns"] + ["Attrition"])
DEFAULTS = df.median(numeric_only=True).to_dict()
DEFAULTS.update({c: df[c].mode()[0] for c in CAT_COLS})


def predict_attrition(employee: dict):
    """Takes employee details and returns the probability of leaving."""
    row = {**DEFAULTS, **employee}
    row = pd.DataFrame([row])
    row = pd.get_dummies(row, columns=[c for c in CAT_COLS if c in row.columns])
    # Arrange the columns in the same order as training, missing ones become 0
    row = row.reindex(columns=FEATURES, fill_value=0).astype("float32")

    prob = float(model.predict(scaler.transform(row.values), verbose=0).ravel()[0])
    if prob >= THRESHOLD:
        risk = "HIGH"
    elif prob >= THRESHOLD * 0.6:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return prob, risk


def ask(question, default, cast=str):
    """Ask the user, and use the default value if they just press Enter."""
    ans = input(f"{question} [{default}]: ").strip()
    return cast(ans) if ans else default


if __name__ == "__main__":
    print("=" * 55)
    print("EMPLOYEE ATTRITION PREDICTION")
    print(f"(press Enter to use the default value shown in brackets)")
    print("=" * 55)

    employee = {
        "Age":              ask("Age", 30, int),
        "MonthlyIncome":    ask("Monthly income", 5000, int),
        "OverTime":         ask("Works overtime (Yes/No)", "No"),
        "JobLevel":         ask("Job level (1-5)", 2, int),
        "YearsAtCompany":   ask("Years at company", 5, int),
        "TotalWorkingYears": ask("Total working years", 10, int),
        "JobSatisfaction":  ask("Job satisfaction (1-4)", 3, int),
        "WorkLifeBalance":  ask("Work life balance (1-4)", 3, int),
        "MaritalStatus":    ask("Marital status (Single/Married/Divorced)", "Single"),
        "DistanceFromHome": ask("Distance from home (km)", 8, int),
    }

    prob, risk = predict_attrition(employee)

    print("\n" + "=" * 55)
    print(f"Probability of leaving : {prob:.1%}")
    print(f"Prediction             : {'Will leave' if prob >= THRESHOLD else 'Will stay'}")
    print(f"Risk level             : {risk}")
    print("=" * 55)
