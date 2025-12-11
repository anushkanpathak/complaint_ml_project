# src/train.py
import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import tokenize_and_lemmatize, normalize_text

DATA_PATH = os.path.join('..','data','complaints.csv')  # from src/
MODEL_OUT = os.path.join('..','models','complaint_pipeline.joblib')

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text','label'])
X = df['text'].values
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # stratify hata diya
)


# Pipeline: TF-IDF with custom tokenizer + classifier options
tfidf = TfidfVectorizer(
    tokenizer=tokenize_and_lemmatize,
    preprocessor=normalize_text,
    ngram_range=(1,2),
    max_features=5000,
    min_df=1
)

# try both models
pipeline_nb = Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())])
pipeline_lr = Pipeline([('tfidf', tfidf), ('clf', LogisticRegression(max_iter=1000))])

print("Training MultinomialNB...")
pipeline_nb.fit(X_train, y_train)
print("Training LogisticRegression...")
pipeline_lr.fit(X_train, y_train)

# Evaluate
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

evaluate(pipeline_nb, X_test, y_test, "MultinomialNB")
evaluate(pipeline_lr, X_test, y_test, "LogisticRegression")

# pick best by accuracy
acc_nb = accuracy_score(y_test, pipeline_nb.predict(X_test))
acc_lr = accuracy_score(y_test, pipeline_lr.predict(X_test))
best = pipeline_lr if acc_lr >= acc_nb else pipeline_nb
print("Best model:", "LogisticRegression" if best is pipeline_lr else "MultinomialNB")

# save
os.makedirs(os.path.join('..','models'), exist_ok=True)
joblib.dump(best, MODEL_OUT)
print("Saved pipeline to", MODEL_OUT)
