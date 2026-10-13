
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("10_Fake_News/dataset.csv")

print("Dataset:")
print(data)

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nActual Labels:")
print(list(y_test))

print("\nPredicted Labels:")
print(list(y_pred))

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["Fake", "Real"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

news = input("\nEnter a news headline to classify: ")
news_tfidf = vectorizer.transform([news])
prediction = model.predict(news_tfidf)[0]

print("Predicted Category:", prediction)
