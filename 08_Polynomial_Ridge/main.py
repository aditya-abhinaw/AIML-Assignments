
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("08_Polynomial_Ridge/dataset.csv")

print("Dataset:")
print(data)

X = data[["Experience"]]
y = data["Salary"]

polynomial_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

ridge_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0)
)

polynomial_model.fit(X, y)
ridge_model.fit(X, y)

polynomial_pred = polynomial_model.predict(X)
ridge_pred = ridge_model.predict(X)

print("\nPolynomial Regression")
print("MSE:", round(mean_squared_error(y, polynomial_pred), 2))
print("R2 Score:", round(r2_score(y, polynomial_pred), 3))

print("\nRidge Regression")
print("MSE:", round(mean_squared_error(y, ridge_pred), 2))
print("R2 Score:", round(r2_score(y, ridge_pred), 3))

plt.scatter(X["Experience"], y, label="Actual Data")
plt.plot(X["Experience"], polynomial_pred, color="red", label="Polynomial Regression")
plt.plot(X["Experience"], ridge_pred, color="green", linestyle="--", label="Ridge Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Polynomial and Ridge Regression")
plt.legend()
plt.grid()
plt.show()

