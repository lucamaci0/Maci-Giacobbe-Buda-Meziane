import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
BASE_COLORS = {
    0:"#1f77b4", 1:"#ff7f0e", 2:"#2ca02c", 3:"#d62728", 4:"#9467bd",
    5:"#8c564b", 6:"#e377c2", 7:"#7f7f7f", 8:"#bcbd22", 9:"#17becf"
}




placeholder = "Online_Retail.csv"
df = pd.read_csv(placeholder, encoding='windows-1252')

df = df.dropna(subset=["CustomerID"])
print(df.columns)
print('Orinal Data')
print(df.describe())
print(df.head())
print(df.isna().sum())

df.dropna(subset=['CustomerID'], inplace=True)

df['InvoiceValue'] = df['Quantity'] * df['UnitPrice']

df_spesa = df.groupby('CustomerID',as_index=False)['InvoiceValue'].sum()

df = df.groupby('CustomerID').agg(
    CustomerLifetimeValue=('InvoiceValue', 'sum'),
    TotalItems=('Quantity', 'sum'), 
    AvgSpesa = ('InvoiceValue', 'mean'),
    # Frequency=('InvoiceDate', lambda x: (x.max()-x.min()).days / x.nunique() if x.nunique()>1 else 1)  # Frequenza acquisti
).reset_index()

print(df.head())


###################################################
# Data Preprocessing Start
###################################################

print("Data Processing...")

# Scaling
feat_for_clustering = ['CustomerLifetimeValue', 'AvgSpesa']  #PLACEHOLDER
feat_scaled = ['SCALED_CustomerLifetimeValue', 'SCALED_AvgSpesa'] #PLACEHOLDER
scaler = StandardScaler()
df[feat_scaled] = scaler.fit_transform(df[feat_for_clustering])
df[feat_scaled] = df[feat_scaled].to_numpy()

print('DESCRIBE')
print(df.describe())
print(df.head())

###################################################
# Elbow graphs with inertia and silhouette
###################################################

# Analyze which n_cluster makes more sense with an elbow-curve approach
ks = range(2, 11)
inertias = []
silhouettes = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(df[feat_scaled])
    # elbow with inertia:
    inertias.append(km.inertia_)
    # elbow with silhouette:
    silhouettes.append(silhouette_score(df[feat_scaled], km.fit_predict(df[feat_scaled])))

ks_sil = ks[1:] if len(silhouettes) == len(ks) - 1 else ks
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
# Right plot: elbow with inertia
axes[0].plot(ks, inertias, marker="o")
axes[0].set_xticks(list(ks))
axes[0].set_xlabel("k (number of clusters)")
axes[0].set_ylabel("Inertia (sum of squared distances)")
axes[0].set_title("Elbow (Inertia vs k)")
axes[0].grid(True, alpha=0.3)
# Right plot: elbow with silhouettes
axes[1].plot(ks_sil, silhouettes, marker="o")
axes[1].set_xticks(list(ks_sil))
axes[1].set_xlabel("k (number of clusters)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette vs k")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "elbow_curves.png"), dpi=300, bbox_inches="tight")

# Silhouettes-Elbow curve shows which be the most reasonable number of clusters for this population.
n_clusters = silhouettes.index(max(silhouettes)) + 2  # index 0 is 2 clusters, index 1 is 3 clusters, etc...


###################################################
# Applying the K-Means Model
###################################################

print(f"\n{n_clusters} seems like the most reasonable number of clusters according to the elbow-graph.")
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
df["km_label"] = kmeans.fit_predict(df[feat_scaled])

# Plot of the clusters
labels = df["km_label"].to_numpy()
X = df[feat_for_clustering].to_numpy()  # original units for nicer axes
centers = scaler.inverse_transform(kmeans.cluster_centers_)

cluster_colors = {int(c): BASE_COLORS[int(c) % 10] for c in np.unique(labels)}
plt.figure(figsize=(6,6))
point_colors = [cluster_colors[int(c)] for c in labels]
plt.scatter(X[:,0], X[:,1], c=point_colors, s=25)
center_colors = [cluster_colors[i] for i in range(n_clusters)] 
plt.scatter(centers[:,0], centers[:,1], c=center_colors, s=200, marker="X", edgecolor="k")
plt.xlabel(feat_for_clustering[0]); plt.ylabel(feat_for_clustering[1])
plt.tight_layout(); 
plt.savefig(os.path.join(PLOTS_DIR, "k-means_clustering.png"), dpi=300, bbox_inches="tight")

# Plot of the clusters (IN SCALED UNITS)
labels = df["km_label"].to_numpy()
X = df[feat_scaled].to_numpy()  # original units for nicer axes
centers = kmeans.cluster_centers_

cluster_colors = {int(c): BASE_COLORS[int(c) % 10] for c in np.unique(labels)}
plt.figure(figsize=(6,6))
point_colors = [cluster_colors[int(c)] for c in labels]
plt.scatter(X[:,0], X[:,1], c=point_colors, s=25)
center_colors = [cluster_colors[i] for i in range(n_clusters)] 
plt.scatter(centers[:,0], centers[:,1], c=center_colors, s=200, marker="X", edgecolor="k")
plt.xlabel(feat_scaled[0]); plt.ylabel(feat_scaled[1])
plt.tight_layout(); 
plt.savefig(os.path.join(PLOTS_DIR, "SCALED_k-means_clustering.png"), dpi=300, bbox_inches="tight")


###################################################
# Evaluating the Silhouette of the clustered points
###################################################

# Silhouette metrics
X_scaled = df[feat_scaled].to_numpy()
labels   = df["km_label"].to_numpy()

sil_avg = float(silhouette_score(X_scaled, labels))
sample_sil = np.array(silhouette_samples(X_scaled, labels))

# Sort by (cluster label, then descending silhouette) for a clean stacked look
order = np.lexsort((-sample_sil, labels))
sorted_scores  = sample_sil[order]
sorted_labels  = labels[order]
bar_colors = [cluster_colors[int(c)] for c in sorted_labels]

plt.figure(figsize=(12, 4))
cmap = cm.get_cmap("tab10", len(np.unique(labels)))

plt.bar(range(len(sorted_scores)), sorted_scores,
        color=bar_colors, edgecolor="black", linewidth=0.3)
plt.axhline(sil_avg, color="red", linestyle="--", linewidth=2,
            label=f"Mean silhouette = {sil_avg:.2f}")
plt.title("Silhouette score per point (colored by KMeans cluster)")
plt.xlabel("Points sorted by cluster")
plt.ylabel("Silhouette score")
plt.xticks([])
plt.ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "silhouette_points.png"), dpi=300, bbox_inches="tight")