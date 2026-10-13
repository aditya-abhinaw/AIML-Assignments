
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("07_SVR/dataset.csv")

print("Dataset:")
print(data)

X = data[["Experience"]]
y = data["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()

model = SVR(kernel="rbf", C=100, epsilon=0.1)
model.fit(X_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

print("\nActual Salaries:")
print(list(y_test))

print("\nPredicted Salaries:")
print(y_pred.round(2))

print("\nMean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

X_plot_scaled = x_scaler.transform(X)
y_plot_scaled = model.predict(X_plot_scaled)
y_plot = y_scaler.inverse_transform(y_plot_scaled.reshape(-1, 1)).ravel()

plt.scatter(X["Experience"], y, label="Actual Data")
plt.plot(X["Experience"], y_plot, color="red", label="SVR Prediction")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Support Vector Regression")
plt.legend()
plt.grid()
plt.show()
