# AIML Assignment 1
# Identify the Association Between Dependent and Independent Variables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create Dataset
# -----------------------------
data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [45, 50, 55, 60, 65, 72, 78, 85, 90]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Define Variables
# -----------------------------
# Independent Variable (X)
X = df["Hours_Studied"].values

# Dependent Variable (Y)
Y = df["Marks"].values

# -----------------------------
# 3. Calculate Correlation
# -----------------------------
correlation = np.corrcoef(X, Y)[0, 1]

# -----------------------------
# 4. Linear Regression
# -----------------------------
slope, intercept = np.polyfit(X, Y, 1)

# Predicted values
predicted_Y = slope * X + intercept

# -----------------------------
# 5. Display Results
# -----------------------------
print("==========================================")
print("       AIML ASSIGNMENT 1")
print("==========================================")

print("\nIndependent Variable : Hours Studied")
print("Dependent Variable   : Marks")

print(f"\nCorrelation Coefficient : {correlation:.2f}")
print(f"Regression Slope        : {slope:.2f}")
print(f"Regression Intercept    : {intercept:.2f}")

print("\nDataset:")
print(df)

# -----------------------------
# 6. Interpret Association
# -----------------------------
if correlation > 0:
    print("\nAssociation: Positive")
elif correlation < 0:
    print("\nAssociation: Negative")
else:
    print("\nAssociation: No Linear Association")

# -----------------------------
# 7. Plot Graph
# -----------------------------
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, predicted_Y, label="Regression Line")

plt.xlabel("Hours Studied (Independent Variable)")
plt.ylabel("Marks (Dependent Variable)")
plt.title("Association Between Study Hours and Marks")

plt.legend()
plt.grid(True)
plt.show()