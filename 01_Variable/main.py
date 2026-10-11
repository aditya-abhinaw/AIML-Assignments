
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("01_Variable/dataset.csv")

print("Dataset:")
print(data)

correlation = data["Hours_Studied"].corr(data["Marks"])
print("\nCorrelation:", round(correlation, 2))

x = data["Hours_Studied"]
y = data["Marks"]

m, c = np.polyfit(x, y, 1)

plt.scatter(x, y)
plt.plot(x, m * x + c)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.grid()
plt.show()
