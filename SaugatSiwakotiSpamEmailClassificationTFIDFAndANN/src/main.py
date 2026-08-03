"""
Spam Email Classification using TF-IDF and an Artificial Neural Network (ANN)
============================================================================

Author : Saugat Siwakoti
Course : Machine Learning (College Project)

What this program does
----------------------
1. Loads a dataset of emails/messages that are labelled as "spam" or "ham".
2. Cleans the text (lowercase, remove punctuation, stopwords, stemming, ...).
3. Converts the cleaned text into numbers using a TF-IDF Vectorizer.
4. Trains a small Artificial Neural Network (ANN) built with Keras.
5. Evaluates the model (accuracy, confusion matrix, precision, recall, F1).
6. Saves the graphs, the classification report and the trained model.
7. Predicts a few custom emails written by hand.

How to run
----------
    python src/main.py

Everything is written in a simple, top-to-bottom style so that each step can be
explained easily during the viva.
"""

# =============================================================================
# STEP 1 : IMPORT THE REQUIRED LIBRARIES
# -----------------------------------------------------------------------------
# What it does : brings in all the tools we need.
# Why needed   : Python does not know about pandas, numpy, ... unless imported.
# Expected     : no output, the program just gets access to these libraries.
# =============================================================================

import os                       # to build file paths and create folders

# -----------------------------------------------------------------------------
# Work from the folder that contains this file.
#
# NLTK version 3.10 and newer refuse to load their own helper libraries when
# those files are found inside the current working directory. If the virtual
# environment is kept in the project folder (a very common setup) this stops
# the program with the message:
#     "ImportError: Blocked import of regex from current working directory"
#
# Switching to the "src" folder first keeps that check happy, and it also means
# the program can be started from anywhere -- from the project folder, from
# inside "src", or with the Run button in an editor such as VS Code.
# This has to happen before NLTK is imported below.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

import re                       # regular expressions, used to clean the text
import string                   # gives us the list of punctuation characters

import numpy as np              # for numerical arrays
import pandas as pd             # for reading and handling the dataset (tables)
import matplotlib               # for plotting graphs
import matplotlib.pyplot as plt

import nltk                                # Natural Language Toolkit
from nltk.corpus import stopwords          # list of common English words
from nltk.stem import PorterStemmer        # cuts words down to their root

from sklearn.feature_extraction.text import TfidfVectorizer   # text -> numbers
from sklearn.model_selection import train_test_split         # train/test split
from sklearn.metrics import (                                # evaluation tools
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

import tensorflow as tf                         # deep learning library
from tensorflow import keras                # easy high-level API of TF
from tensorflow.keras.models import Sequential  # a stack of layers
from tensorflow.keras.layers import Dense, Dropout, Input


# -----------------------------------------------------------------------------
# Draw the graphs straight to a file instead of opening a pop-up window.
# This must be done before any graph is created, and it lets the program run
# on a computer that has no screen (for example a server).
# -----------------------------------------------------------------------------
matplotlib.use("Agg")


# -----------------------------------------------------------------------------
# Make the results reproducible.
#
# Neural networks start with random weights, so every run would normally give
# slightly different numbers. Two things are needed to stop that:
#
# 1. set_random_seed(42) fixes the "seed" of the random number generators, so
#    the train/test split and the starting weights are always the same.
# 2. enable_op_determinism() forces TensorFlow to always add up the numbers in
#    the same order. Without it the accuracy still moves by about 0.1 - 0.3 %
#    between runs, because the CPU splits the work over several cores and the
#    order in which the results come back can change.
#
# Together they mean this program prints exactly the same results every time.
# -----------------------------------------------------------------------------
np.random.seed(42)
keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


# -----------------------------------------------------------------------------
# File paths.
# The script lives inside "src/", so we go one folder up to reach the project
# root and from there we can reach dataset/, model/ and results/.
# -----------------------------------------------------------------------------
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_FOLDER, "dataset", "mail_data.csv")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "model")

# Create the output folders if they do not exist yet.
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# -----------------------------------------------------------------------------
# NLTK needs a few small data files: the stopword list and the tokenizer.
# We download them only once; if they are already present nothing happens.
# -----------------------------------------------------------------------------
print("Downloading the NLTK data files (only needed the first time)...")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
print("NLTK data is ready.\n")


# =============================================================================
# STEP 2 : LOAD THE DATASET
# -----------------------------------------------------------------------------
# What it does : reads the CSV file into a pandas DataFrame (a table).
# Why needed   : we cannot train a model without data.
# Expected     : a table with two columns -> label and text.
# =============================================================================

print("=" * 70)
print("STEP 2 : LOADING THE DATASET")
print("=" * 70)

email_data = pd.read_csv(DATASET_PATH)

# The downloaded CSV file names its columns "Category" and "Message".
# We rename them to "label" and "text" because those names are easier to read.
email_data = email_data.rename(
    columns={"Category": "label", "Message": "text"}
)

print("Dataset loaded from :", DATASET_PATH)
print("Column names        :", list(email_data.columns))
print()


# =============================================================================
# STEP 3 : EXPLORE THE DATASET
# -----------------------------------------------------------------------------
# What it does : prints basic information about the data.
# Why needed   : before training we must know how much data we have, whether
#                any value is missing and how many spam / ham messages exist.
# Expected     : shape, first five rows, missing values and class distribution.
# =============================================================================

print("=" * 70)
print("STEP 3 : EXPLORING THE DATASET")
print("=" * 70)

# 3.1 Shape -> (number of rows, number of columns)
print("Dataset shape (rows, columns):", email_data.shape)
print()

# 3.2 First five rows -> a quick look at what the data actually contains
print("First five rows of the dataset:")
print(email_data.head())
print()

# 3.3 Missing values -> empty cells would break the preprocessing step
print("Missing values in each column:")
print(email_data.isnull().sum())
print()

# If any row is empty we simply remove it (there are usually none).
email_data = email_data.dropna()

# 3.4 Class distribution -> how many "ham" and how many "spam" messages
print("Class distribution (how many messages of each type):")
print(email_data["label"].value_counts())
print()
print("Class distribution in percentage:")
print(round(email_data["label"].value_counts(normalize=True) * 100, 2))
print()

# 3.5 Convert the text labels into numbers, because a neural network can only
#     work with numbers:  ham -> 0 (not spam)  and  spam -> 1 (spam)
email_data["label_number"] = email_data["label"].map({"ham": 0, "spam": 1})
print("Labels converted to numbers -> ham = 0, spam = 1")
print(email_data[["label", "label_number", "text"]].head())
print()


# =============================================================================
# STEP 4 : TEXT PREPROCESSING
# -----------------------------------------------------------------------------
# What it does : cleans one email at a time.
# Why needed   : raw text contains punctuation, numbers and very common words
#                ("the", "is", "and") that do not help to detect spam. Removing
#                them makes the data smaller and the model more accurate.
# Expected     : a short, clean sentence made only of important root words.
# =============================================================================

print("=" * 70)
print("STEP 4 : TEXT PREPROCESSING")
print("=" * 70)

# These two objects are created only once (outside the function) because
# building them again for every email would make the program very slow.
english_stopwords = set(stopwords.words("english"))
porter_stemmer = PorterStemmer()


def clean_text(email_text):
    """Clean a single email and return it as a simple cleaned sentence.

    Steps: lowercase -> remove punctuation -> remove numbers ->
    remove extra spaces -> tokenize -> remove stopwords -> Porter stemming.
    """

    # 4.1 Lowercase conversion.
    #     "FREE" and "free" mean the same thing, so we make everything small.
    email_text = email_text.lower()

    # 4.2 Remove punctuation such as . , ! ? $ % etc.
    #     They carry almost no meaning for our model.
    email_text = email_text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # 4.3 Remove numbers (phone numbers, prices, ...).
    #     \d means "any digit" and + means "one or more of them".
    email_text = re.sub(r"\d+", " ", email_text)

    # 4.4 Remove extra spaces, tabs and newlines, keeping a single space.
    email_text = re.sub(r"\s+", " ", email_text).strip()

    # 4.5 Tokenize -> break the sentence into a list of separate words.
    #     "win free money" becomes ["win", "free", "money"]
    word_list = nltk.word_tokenize(email_text)

    # 4.6 Remove English stopwords -> very common words like "the", "is", "at".
    #     Words of length 1 (leftovers such as "u" split by cleaning) are
    #     also dropped because they are not useful.
    word_list = [
        word for word in word_list
        if word not in english_stopwords and len(word) > 1
    ]

    # 4.7 Porter Stemming -> cut every word down to its root form so that
    #     "winning", "wins" and "winner" all become "win".
    word_list = [porter_stemmer.stem(word) for word in word_list]

    # 4.8 Join the words back into one sentence, because the TF-IDF Vectorizer
    #     expects a sentence (a string), not a list of words.
    cleaned_text = " ".join(word_list)

    return cleaned_text


# Apply the cleaning function to every single email in the dataset.
# .apply() simply runs our function once for each row.
print("Cleaning all the emails, please wait a few seconds...")
email_data["cleaned_text"] = email_data["text"].apply(clean_text)
print("Cleaning finished.\n")

# Show one example so we can see the difference between before and after.
print("Example of preprocessing")
print("-" * 70)
print("ORIGINAL :", email_data["text"].iloc[2][:120])
print("CLEANED  :", email_data["cleaned_text"].iloc[2][:120])
print()


# =============================================================================
# STEP 5 : TF-IDF FEATURE EXTRACTION
# -----------------------------------------------------------------------------
# What it does : turns every cleaned email into a row of numbers.
# Why needed   : a neural network understands numbers, not words.
# How it works : TF (Term Frequency) counts how often a word appears in an
#                email, IDF (Inverse Document Frequency) lowers the score of
#                words that appear in almost every email. Rare but meaningful
#                words such as "winner" or "prize" therefore get a high score.
# Expected     : a matrix of shape (number of emails, 3000 features).
# =============================================================================

print("=" * 70)
print("STEP 5 : TF-IDF FEATURE EXTRACTION")
print("=" * 70)

# max_features=3000 keeps only the 3000 most useful words. This keeps the
# model small and fast, which is enough for a college project.
tfidf_vectorizer = TfidfVectorizer(max_features=3000)

# fit_transform() learns the vocabulary AND converts the text in one call.
feature_matrix = tfidf_vectorizer.fit_transform(email_data["cleaned_text"])

# The result is a sparse matrix (it stores mostly zeros in a compact way).
# Keras needs a normal dense NumPy array, so we convert it.
X = feature_matrix.toarray()

# y holds the answers we want the model to learn (0 = ham, 1 = spam).
y = email_data["label_number"].values

print("Shape of the feature matrix X :", X.shape)
print("Shape of the label vector  y :", y.shape)
print("Number of words in vocabulary :", len(tfidf_vectorizer.vocabulary_))
print()


# =============================================================================
# STEP 6 : SPLIT THE DATASET (80 % TRAINING, 20 % TESTING)
# -----------------------------------------------------------------------------
# What it does : divides the data into a training part and a testing part.
# Why needed   : the model must be tested on emails it has never seen, exactly
#                like an exam tests a student on new questions.
# Expected     : about 4457 training emails and 1115 testing emails.
# =============================================================================

print("=" * 70)
print("STEP 6 : SPLITTING THE DATASET")
print("=" * 70)

# test_size=0.2   -> 20 % of the data goes to testing
# random_state=42 -> the same split every time we run the program
# stratify=y      -> keeps the same spam/ham proportion in both parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])
print("Spam messages in the training set :", int(y_train.sum()))
print("Spam messages in the testing set  :", int(y_test.sum()))
print()


# =============================================================================
# STEP 7 : BUILD THE ARTIFICIAL NEURAL NETWORK (ANN)
# -----------------------------------------------------------------------------
# What it does : creates the layers of the neural network.
# Why needed   : this network is the "brain" that learns to separate spam
#                from ham.
# Architecture : Input -> Dense(128, relu) -> Dropout(0.3)
#                      -> Dense(64, relu)  -> Dense(1, sigmoid)
# Expected     : a printed summary of the model with its layers and parameters.
# =============================================================================

print("=" * 70)
print("STEP 7 : BUILDING THE ANN MODEL")
print("=" * 70)


def build_ann_model(number_of_features):
    """Create and compile the ANN used in this project."""

    model = Sequential([
        # Input layer: tells Keras how many numbers each email has (3000).
        Input(shape=(number_of_features,)),

        # Hidden layer 1: 128 neurons.
        # 'relu' keeps positive values and turns negative values into 0,
        # which helps the network learn non-linear patterns.
        Dense(128, activation="relu"),

        # Dropout: during training it randomly switches off 30 % of the
        # neurons. This prevents the model from simply memorising the
        # training emails (a problem called overfitting).
        Dropout(0.3),

        # Hidden layer 2: 64 neurons, learns more compact patterns.
        Dense(64, activation="relu"),

        # Output layer: a single neuron with 'sigmoid'.
        # Sigmoid squeezes the result between 0 and 1, so it can be read
        # as "the probability that this email is spam".
        Dense(1, activation="sigmoid"),
    ])

    # Compile = choose how the model will learn.
    # optimizer='adam'                -> a good, automatic learning rate method
    # loss='binary_crossentropy'      -> the standard loss for 2-class problems
    # metrics=['accuracy']            -> we want to watch the accuracy
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


ann_model = build_ann_model(X_train.shape[1])

# Print the structure of the network so we can explain it during the viva.
ann_model.summary()
print()


# =============================================================================
# STEP 8 : TRAIN THE MODEL
# -----------------------------------------------------------------------------
# What it does : shows the training emails to the network 10 times (10 epochs).
# Why needed   : this is the real learning step, where the weights change.
# Parameters   : batch_size=32       -> 32 emails are processed at a time
#                validation_split=0.2-> 20 % of the training data is kept aside
#                                       to check the model while it learns
# Expected     : accuracy going up and loss going down at every epoch.
# =============================================================================

print("=" * 70)
print("STEP 8 : TRAINING THE ANN (10 EPOCHS)")
print("=" * 70)

training_history = ann_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

print("\nTraining completed.\n")


# =============================================================================
# STEP 9 : EVALUATE THE MODEL ON THE TEST SET
# -----------------------------------------------------------------------------
# What it does : checks how well the model works on emails it never saw.
# Why needed   : this is the honest score of our project.
# Expected     : accuracy, confusion matrix, classification report,
#                precision, recall and F1 score.
# =============================================================================

print("=" * 70)
print("STEP 9 : EVALUATING THE MODEL")
print("=" * 70)

# 9.1 The model outputs a probability between 0 and 1 for every test email.
predicted_probabilities = ann_model.predict(X_test, verbose=0)

# 9.2 We turn the probability into a final answer:
#     probability >= 0.5 -> spam (1),  probability < 0.5 -> ham (0)
predicted_labels = (predicted_probabilities >= 0.5).astype(int).flatten()

# 9.3 Accuracy = how many predictions were correct out of all predictions.
test_accuracy = accuracy_score(y_test, predicted_labels)
print("Test Accuracy :", round(test_accuracy * 100, 2), "%")
print()

# 9.4 Confusion matrix.
#     [[True Negative , False Positive]
#      [False Negative, True Positive ]]
#     It shows exactly which kind of mistakes the model makes.
confusion = confusion_matrix(y_test, predicted_labels)
print("Confusion Matrix:")
print(confusion)
print()
print("Reading the confusion matrix:")
print("  True  Negatives (ham  predicted as ham ) :", confusion[0][0])
print("  False Positives (ham  predicted as spam) :", confusion[0][1])
print("  False Negatives (spam predicted as ham ) :", confusion[1][0])
print("  True  Positives (spam predicted as spam) :", confusion[1][1])
print()

# 9.5 Precision, Recall and F1 score for the spam class.
#     Precision = of all emails we called spam, how many really were spam.
#     Recall    = of all real spam emails, how many did we catch.
#     F1 score  = a single balanced number combining precision and recall.
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

print("Precision (spam) :", round(precision, 4))
print("Recall    (spam) :", round(recall, 4))
print("F1 Score  (spam) :", round(f1, 4))
print()

# 9.6 The classification report shows all of the above for both classes.
report_text = classification_report(
    y_test, predicted_labels, target_names=["ham", "spam"]
)
print("Classification Report:")
print(report_text)

# 9.7 Save everything into results/classification_report.txt so that the
#     numbers can be copied into the project report later.
report_file_path = os.path.join(RESULTS_FOLDER, "classification_report.txt")

with open(report_file_path, "w") as report_file:
    report_file.write("SPAM EMAIL CLASSIFICATION USING TF-IDF AND ANN\n")
    report_file.write("=" * 60 + "\n\n")
    accuracy_percent = round(test_accuracy * 100, 2)
    report_file.write("Test Accuracy : " + str(accuracy_percent) + " %\n")
    report_file.write("Precision     : " + str(round(precision, 4)) + "\n")
    report_file.write("Recall        : " + str(round(recall, 4)) + "\n")
    report_file.write("F1 Score      : " + str(round(f1, 4)) + "\n\n")
    report_file.write("Confusion Matrix\n")
    report_file.write("-" * 60 + "\n")
    report_file.write(str(confusion) + "\n\n")
    report_file.write("True  Negatives : " + str(confusion[0][0]) + "\n")
    report_file.write("False Positives : " + str(confusion[0][1]) + "\n")
    report_file.write("False Negatives : " + str(confusion[1][0]) + "\n")
    report_file.write("True  Positives : " + str(confusion[1][1]) + "\n\n")
    report_file.write("Classification Report\n")
    report_file.write("-" * 60 + "\n")
    report_file.write(report_text + "\n")

print("Classification report saved to :", report_file_path)
print()


# =============================================================================
# STEP 10 : PLOT THE GRAPHS
# -----------------------------------------------------------------------------
# What it does : draws accuracy/loss curves and the confusion matrix.
# Why needed   : graphs make it easy to see whether the model learned well
#                or whether it overfitted.
# Expected     : two PNG images inside the results/ folder.
# =============================================================================

print("=" * 70)
print("STEP 10 : SAVING THE GRAPHS")
print("=" * 70)

# --- Graph 1 : training / validation accuracy and loss -----------------------
# training_history.history is a dictionary that stored the numbers of every
# epoch while the model was training.
figure, (accuracy_plot, loss_plot) = plt.subplots(1, 2, figsize=(13, 5))

# Left plot: accuracy
accuracy_plot.plot(training_history.history["accuracy"], marker="o",
                   label="Training Accuracy")
accuracy_plot.plot(training_history.history["val_accuracy"], marker="o",
                   label="Validation Accuracy")
accuracy_plot.set_title("Training vs Validation Accuracy")
accuracy_plot.set_xlabel("Epoch")
accuracy_plot.set_ylabel("Accuracy")
accuracy_plot.legend()
accuracy_plot.grid(True, linestyle="--", alpha=0.5)

# Right plot: loss
loss_plot.plot(training_history.history["loss"], marker="o",
               label="Training Loss")
loss_plot.plot(training_history.history["val_loss"], marker="o",
               label="Validation Loss")
loss_plot.set_title("Training vs Validation Loss")
loss_plot.set_xlabel("Epoch")
loss_plot.set_ylabel("Loss")
loss_plot.legend()
loss_plot.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
accuracy_loss_path = os.path.join(RESULTS_FOLDER, "accuracy_loss_graph.png")
plt.savefig(accuracy_loss_path, dpi=120)
plt.close()
print("Accuracy / Loss graph saved to :", accuracy_loss_path)

# --- Graph 2 : confusion matrix as a coloured picture ------------------------
plt.figure(figsize=(6, 5))
plt.imshow(confusion, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

class_names = ["ham", "spam"]
plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Write the four numbers inside the coloured squares.
for row in range(2):
    for column in range(2):
        plt.text(
            column,
            row,
            str(confusion[row][column]),
            ha="center",
            va="center",
            fontsize=16,
            color="red",
        )

plt.tight_layout()
confusion_matrix_path = os.path.join(RESULTS_FOLDER, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=120)
plt.close()
print("Confusion matrix image saved to :", confusion_matrix_path)
print()


# =============================================================================
# STEP 11 : SAVE THE TRAINED MODEL
# -----------------------------------------------------------------------------
# What it does : writes the trained network to a file.
# Why needed   : so we do not have to train it again next time; the file can
#                be loaded later with keras.models.load_model().
# Expected     : model/spam_classifier.keras
# =============================================================================

print("=" * 70)
print("STEP 11 : SAVING THE TRAINED MODEL")
print("=" * 70)

model_path = os.path.join(MODEL_FOLDER, "spam_classifier.keras")
ann_model.save(model_path)
print("Trained model saved to :", model_path)
print()


# =============================================================================
# STEP 12 : PREDICT CUSTOM EMAIL EXAMPLES
# -----------------------------------------------------------------------------
# What it does : tests the model on five emails written by hand.
# Why needed   : it proves that the model really works on new text and it is
#                the easiest part to demonstrate during the viva.
# Expected     : the email, the probability and the final Spam / Ham decision.
# =============================================================================

print("=" * 70)
print("STEP 12 : PREDICTING CUSTOM EMAILS")
print("=" * 70)


def predict_email(email_text):
    """Predict whether one new email is Spam or Ham.

    Returns the predicted label ("Spam"/"Ham") and the spam probability.
    """

    # 12.1 Clean the new email exactly the same way as the training data.
    #      If we skip this the words would not match the vocabulary.
    cleaned_email = clean_text(email_text)

    # 12.2 Convert it into TF-IDF numbers.
    #      Note: we use transform(), NOT fit_transform(), because the
    #      vectorizer already learned the vocabulary from the training data.
    email_features = tfidf_vectorizer.transform([cleaned_email]).toarray()

    # 12.3 Ask the trained model for a probability.
    prediction = ann_model.predict(email_features, verbose=0)
    spam_probability = float(prediction[0][0])

    # 12.4 Convert the probability into a readable answer.
    if spam_probability >= 0.5:
        predicted_label = "Spam"
    else:
        predicted_label = "Ham"

    return predicted_label, spam_probability


# Five (plus a few extra) hand written test emails.
custom_emails = [
    "Congratulations! You have WON a FREE iPhone 15. Click this link now to claim your prize!",
    "Hi Saugat, please find attached the notes for tomorrow's machine learning class.",
    "URGENT! Your account has been suspended. Send your password and bank details immediately.",
    "Can we reschedule our project meeting to Friday at 3 pm? Let me know if that works.",
    "WINNER!! You have been selected to receive a cash prize of $50000. Reply YES to claim.",
    "Dear sir, I have submitted the assignment on the college portal. Kindly check it.",
    "FREE entry in our weekly competition to win an iPad. Text the word WIN to 80086 now!",
]

for email_number, email_text in enumerate(custom_emails, start=1):
    predicted_label, spam_probability = predict_email(email_text)

    print("Example", email_number)
    print("-" * 70)
    print("Email          :", email_text)
    print("Spam Probability:", round(spam_probability, 4))
    print("Prediction     :", predicted_label)
    print()


# =============================================================================
# END OF THE PROGRAM
# =============================================================================

print("=" * 70)
print("PROJECT FINISHED SUCCESSFULLY")
print("=" * 70)
print("Final Test Accuracy :", round(test_accuracy * 100, 2), "%")
print("Files created:")
print("  ", model_path)
print("  ", accuracy_loss_path)
print("  ", confusion_matrix_path)
print("  ", report_file_path)
