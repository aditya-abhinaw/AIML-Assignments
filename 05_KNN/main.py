
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("05_KNN/dataset.csv")

print("Dataset:")
print(data)

X = data[["StudyHours", "Attendance"]]
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Results:")
print(list(y_test))

print("\nPredicted Results:")
print(list(y_pred))

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

plt.scatter(data["StudyHours"], data["Attendance"], c=(y == "Pass").astype(int))
plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("KNN Student Classification")
plt.grid()
plt.show()
