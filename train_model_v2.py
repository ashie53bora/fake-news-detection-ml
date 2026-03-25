import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------
# Download NLTK resources
# ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(["reuters", "breaking", "exclusive"])

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ---------------------------
# Load Dataset 1 — Fake.csv / True.csv
# ---------------------------
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

df1 = pd.concat([fake, true], axis=0)
df1["content"] = df1["title"].fillna('') + " " + df1["text"].fillna('')
df1 = df1[["content", "label"]]

print(f"Dataset 1 loaded: {df1.shape[0]} rows")

# ---------------------------
# Load Dataset 2 — fake_or_real_news.csv
# ---------------------------
df2_raw = pd.read_csv("dataset/fake_or_real_news.csv")

# Convert FAKE/REAL text labels to 0/1
df2_raw["label"] = df2_raw["label"].map({"FAKE": 0, "REAL": 1})
df2_raw["content"] = df2_raw["title"].fillna('') + " " + df2_raw["text"].fillna('')
df2 = df2_raw[["content", "label"]]

print(f"Dataset 2 loaded: {df2.shape[0]} rows")

# Load extra balanced data
df3 = pd.read_csv("dataset/extra_data.csv")
df3 = df3[["content", "label"]]
print(f"Extra data loaded: {df3.shape[0]} rows")

# ---------------------------
# Merge Both Datasets
# ---------------------------
df = pd.concat([df1, df2, df3], axis=0).dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total merged rows: {df.shape[0]}")
print(f"Fake: {(df['label']==0).sum()} | Real: {(df['label']==1).sum()}")

# ---------------------------
# Clean Text
# ---------------------------
print("Cleaning text... (this may take a few minutes)")
df["content"] = df["content"].apply(clean_text)
print("Text preprocessing completed!")

# ---------------------------
# TF-IDF Feature Extraction
# ---------------------------
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = tfidf.fit_transform(df["content"])
y = df["label"]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
print("Training model...")

base_model = LogisticRegression(
    max_iter=2000,
    C=1.0,
    solver='liblinear'
)

model = CalibratedClassifierCV(base_model)
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed!")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# ---------------------------
# Quick Test
# ---------------------------
print("\nQuick test predictions:")
tests = [
    ("NASA launches new rocket to the moon", "REAL"),
    ("Apple reports record quarterly earnings", "REAL"),
    ("Scientists confirm 5G causes mind control", "FAKE"),
    ("Aliens land in New York City government confirms", "FAKE"),
]

for text, expected in tests:
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    print(f"  [{expected}] → [{label}] | Fake:{prob[0]*100:.1f}% Real:{prob[1]*100:.1f}% — '{text}'")

# ---------------------------
# Save Model
# ---------------------------
joblib.dump(model, "my_model.pkl")
joblib.dump(tfidf, "my_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")