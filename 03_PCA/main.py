
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("03_PCA/dataset.csv")

print("Original Dataset:")
print(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

result = pd.DataFrame(
    reduced_data,
    columns=["PC1", "PC2"]
)

print("\nReduced Dataset:")
print(result.round(2))

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_.round(3))

print("\nTotal Variance Retained:")
print(round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

plt.scatter(result["PC1"], result["PC2"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Dimensionality Reduction")
plt.grid()
plt.show()
