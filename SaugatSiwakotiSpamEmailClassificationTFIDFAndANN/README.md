# SaugatSiwakotiSpamEmailClassificationTFIDFAndANN

## Student

Name: Saugat Siwakoti

## Project Title

Spam Email Classification using TF-IDF and an Artificial Neural Network (ANN)

## Objective

The objective of this project is to build an Artificial Neural Network (ANN) that
classifies an email or text message as either **spam** (unwanted) or **ham**
(genuine), based only on the words the message contains.

## Problem Statement

Spam messages waste time, fill up the inbox and are often used for phishing and
fraud. Filtering them by hand is impossible because of the huge number of messages
sent every day, and fixed keyword rules are easy for spammers to avoid. The extra
difficulty is that a computer cannot read words — it can only learn from numbers.
This project solves the task as a binary classification problem: the message text
is cleaned using standard NLP steps, converted into numeric features with TF-IDF,
and classified by a fully-connected ANN built with TensorFlow/Keras.

## Dataset

- **Dataset Name:** SMS Spam Collection Dataset
- **Source:** UCI Machine Learning Repository / Kaggle — stored locally as `dataset/mail_data.csv`
- **Total Samples:** 5572
- **Columns:** `Category` (label) and `Message` (text), renamed to `label` and `text` in the code
- **Classes:** ham 4825 (86.59 %), spam 747 (13.41 %) — **imbalanced**
- **Missing values:** none

Because the dataset is imbalanced, a model that always answered "ham" would
already be 86.59 % accurate. Accuracy alone is therefore not enough, and
precision, recall and F1 for the spam class are reported as well.

**Preprocessing:**

- Checked dataset shape, missing values and class distribution
- Encoded the labels as numbers (ham = 0, spam = 1)
- Cleaned every message: lowercase → remove punctuation → remove numbers →
  remove extra spaces → tokenize → remove English stopwords → Porter stemming
- Converted the cleaned text into 3000 numeric features using **TF-IDF**
  (`max_features=3000`)
- 80/20 stratified train-test split (4457 training, 1115 testing messages)

## ANN Architecture

A simple feedforward ANN built with TensorFlow/Keras:

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 3000 (TF-IDF features) | — |
| Hidden 1 | 128 | ReLU |
| Dropout | — (rate 0.3) | — |
| Hidden 2 | 64 | ReLU |
| Output | 1 | Sigmoid |

Trained with the Adam optimizer, binary crossentropy loss, batch size 32, for 10
epochs with a 20 % validation split. Total trainable parameters: 392,449.

## Results

### Model Performance

- **Test Accuracy: 97.67 %**
- Precision (spam): 0.9695 | Recall (spam): 0.8523 | F1 (spam): 0.9071

Training accuracy climbs to 100 % while validation accuracy settles near 97 %.
The validation loss reaches its minimum at about epoch 3 and then slowly rises,
which shows mild overfitting — the reason the `Dropout(0.3)` layer is included.

**Training Curves:**

![Training Curves](results/accuracy_loss_graph.png)

**Confusion Matrix:**

![Confusion Matrix](results/confusion_matrix.png)

Out of 1115 unseen messages the model made only 26 mistakes. Just **4** genuine
messages were wrongly marked as spam, while **22** spam messages slipped through
into the inbox.

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| ham | 0.98 | 1.00 | 0.99 |
| spam | 0.97 | 0.85 | 0.91 |

**Sample Predictions** (new, hand written messages):

| Email | Spam Probability | Predicted Class |
|-------|------------------|-----------------|
| Congratulations! You have WON a FREE iPhone 15. Click this link now to claim your prize! | 1.0000 | Spam |
| Hi Saugat, please find attached the notes for tomorrow's machine learning class. | 0.0006 | Ham |
| URGENT! Your account has been suspended. Send your password and bank details immediately. | 0.9831 | Spam |
| Can we reschedule our project meeting to Friday at 3 pm? Let me know if that works. | 0.0000 | Ham |
| FREE entry in our weekly competition to win an iPad. Text the word WIN to 80086 now! | 0.8434 | Spam |

All seven custom messages in `main.py` were classified correctly, and the model is
confident: spam scored above 0.84 and genuine messages below 0.001.

## Conclusion

This project implemented an end-to-end ANN pipeline for spam detection: data
loading and exploration, NLP text cleaning with stopword removal and Porter
stemming, TF-IDF feature extraction, a stratified train/test split, a
2-hidden-layer ANN with dropout, and full evaluation with graphs. The model
achieves **97.67 % test accuracy** with a spam F1 score of 0.91.

Precision for spam is high (0.97), so genuine emails are almost never lost — the
most harmful mistake a spam filter can make. Recall is lower (0.85), meaning about
15 of every 100 spam messages still reach the inbox. This is a direct effect of the
class imbalance: the network saw 6.5 times more ham than spam, so it leans towards
predicting "ham" when unsure.

Possible improvements: use `class_weight` or oversampling to raise spam recall,
add early stopping to cut off training at the validation-loss minimum, and compare
against classical baselines (Naive Bayes, Logistic Regression, SVM) on the same
TF-IDF features.

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib

## How to Run

```bash
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install pandas numpy matplotlib nltk scikit-learn tensorflow

python src/main.py
```

The script trains the model and saves `model/spam_classifier.keras`, both graphs
and `results/classification_report.txt`. NLTK data downloads automatically on the
first run.

**Important — activate the virtual environment first.** If you see
`ModuleNotFoundError: No module named 'pandas'`, the environment is not active.
In VS Code, press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the one
inside `venv/`, otherwise the Run button uses the system Python, which has none of
these libraries installed.

The script can be started from anywhere — the project folder, from inside `src/`,
or with the Run button in VS Code. It works out its own file paths, so the dataset
and the output folders are always found.

> **Tip:** TensorFlow prints a few `I0000`/`E0000` startup lines. They are harmless.
> Use `TF_CPP_MIN_LOG_LEVEL=3 python src/main.py` for clean output during a demo.

The results are fully reproducible: `main.py` fixes the random seeds **and** calls
`tf.config.experimental.enable_op_determinism()`, so every run prints exactly the
numbers shown above.

A full write-up is available in [report/Project_Report.md](report/Project_Report.md).
