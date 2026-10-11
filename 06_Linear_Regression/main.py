
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("06_Linear_Regression/dataset.csv")

print("Dataset:")
print(data)

X = data[["Hours_Studied"]]
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Marks:")
print(list(y_test))

print("\nPredicted Marks:")
print(y_pred.round(2))

print("\nSlope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

plt.scatter(X, y)
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.grid()
plt.show()
