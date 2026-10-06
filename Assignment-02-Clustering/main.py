import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the dataset
data = pd.read_csv("Assignment-02-Clustering/dataset.csv")

print("Dataset:")
print(data)

# Taking two columns for clustering
x = data[["AnnualIncome", "SpendingScore"]]

# Creating K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Finding clusters
data["Cluster"] = model.fit_predict(x)

print("\nClustered Data:")
print(data)

# Printing cluster centers
print("\nCluster Centers:")
print(model.cluster_centers_)

# Plotting the clusters
plt.scatter(
    data["AnnualIncome"],
    data["SpendingScore"],
    c=data["Cluster"]
)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Clustering using K-Means")
plt.grid()
plt.show()