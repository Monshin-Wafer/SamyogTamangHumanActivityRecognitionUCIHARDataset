import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load ANN model
model = load_model("../model/spam_ann.keras")

# Load vectorizer
vectorizer = joblib.load("../model/vectorizer.pkl")

# User input
message = input("Enter a message: ")

# Convert text to TF-IDF
message_vector = vectorizer.transform([message]).toarray()

# Predict
prediction = model.predict(message_vector)

print("Probability:", prediction[0][0])

if prediction[0][0] >= 0.5:
    print("Prediction: Spam")
else:
    print("Prediction: Ham")