from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 1. Read the dataset (dynamically located relative to this file)
dataset_path = Path(__file__).parent / "dataset.csv"
data = pd.read_csv(dataset_path)

print("Dataset:")
print(data)

# 2. Select features for clustering
X = data[["AnnualIncome", "SpendingScore"]]

# 3. Create K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# 4. Predict clusters
data["Cluster"] = model.fit_predict(X)

print("\nClustered Data:")
print(data)

# 5. Print cluster centers
print("\nCluster Centers:")
print(model.cluster_centers_)

# 6. Plot the clusters and centroids
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    data["AnnualIncome"],
    data["SpendingScore"],
    c=data["Cluster"],
    cmap="viridis",
    label="Customers",
)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolor="black",
    label="Centroids",
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Clustering using K-Means")
plt.legend()
plt.grid(True)
plt.show()