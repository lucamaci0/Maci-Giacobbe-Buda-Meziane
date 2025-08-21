import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
BASE_COLORS = {
    0:"#1f77b4", 1:"#ff7f0e", 2:"#2ca02c", 3:"#d62728", 4:"#9467bd",
    5:"#8c564b", 6:"#e377c2", 7:"#7f7f7f", 8:"#bcbd22", 9:"#17becf"
}
dataset_path = f"{CURRENT_DIR}/datasets/Student_performance_data_.csv"
pd.set_option("display.max_columns", None)


df = pd.read_csv(dataset_path)

print("The shape of the datset is:  ", df.shape)
 
# check for null values
print(df.isnull().sum())
df = df.dropna()  # remove rows with missing values

# check for duplicate rows
print(df.duplicated().sum())
df = df.dropna()  # remove rows with missing values

df = df.reset_index(drop=True)  # reset the index after dropping rows
 
# Display the cleaned DataFrame
print(df.describe())








######################################################################

# Scaling data to improve distance-based clustering
feat_for_clustering = ['StudyTimeWeekly', 'GPA', 'Absences', 'Tutoring'] 
feat_scaled = [f'SCALED_{feat}' for feat in feat_for_clustering] 
feat_to_plot = ['StudyTimeWeekly', 'Tutoring', 'Absences']


scaler = StandardScaler()
df[feat_scaled] = scaler.fit_transform(df[feat_for_clustering])
df[feat_scaled] = df[feat_scaled].to_numpy()

pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(df[feat_scaled])


######################################################################





# Analyze which n_cluster makes more sense with an elbow-curve approach
max_clusters_to_try = 10
ks = range(2, max_clusters_to_try+1)
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
axes[0].set_xlabel("Number of clusters")
axes[0].set_ylabel("Inertia (sum of squared distances)")
axes[0].set_title("Inertia vs n_clusters")
axes[0].grid(True, alpha=0.3)
# Right plot: elbow with silhouettes
axes[1].plot(ks_sil, silhouettes, marker="o")
axes[1].set_xticks(list(ks_sil))
axes[1].set_xlabel("Number of clusters")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette vs n_clusters")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "elbow_curves.png"), dpi=300, bbox_inches="tight")


# Silhouettes-Elbow curve shows which be the most reasonable number of clusters for this population.
best_clusters_silhouettes = sorted(
    [(i + 2, s) for i, s in enumerate(silhouettes)],
    key=lambda t: t[1],          # sort by the score
    reverse=True                 # highest first
)

amount_of_cluster_plots = 2
for ii in range(amount_of_cluster_plots):

  n_clusters = best_clusters_silhouettes[ii][0]
  print(f"\n{n_clusters} seems like the most reasonable number of clusters according to the elbow-graph.")
  kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
  df["km_label"] = kmeans.fit_predict(df[feat_scaled])

  # Plot of the clusters
  labels = df["km_label"].to_numpy()
  X_to_plot = df[feat_to_plot].to_numpy()  # original units for nicer axes
  centers = scaler.inverse_transform(kmeans.cluster_centers_)
  cluster_colors = {int(c): BASE_COLORS[int(c) % 10] for c in np.unique(labels)}

#fix?
#########################
  centers_orig_df = pd.DataFrame(
      scaler.inverse_transform(kmeans.cluster_centers_),
      columns=feat_for_clustering
  )
  C_to_plot = centers_orig_df[feat_to_plot].to_numpy()
#########################

  
  if len(feat_to_plot) >= 3:
      # -------- 3D PLOT --------
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")

    # points
    point_colors = [cluster_colors[int(c)] for c in labels]
    ax.scatter(X_to_plot[:, 0], X_to_plot[:, 1], X_to_plot[:, 2], c=point_colors, s=20, depthshade=False)

    # centers

    center_colors = [cluster_colors[i] for i in range(n_clusters)]
    ax.scatter(C_to_plot[:, 0], C_to_plot[:, 1], C_to_plot[:, 2],             # <<< CHANGED (use C_to_plot, not raw centers[:, :3])
               c=center_colors, s=200, marker="X", edgecolor="k")
    """
    center_colors = [cluster_colors[i] for i in range(n_clusters)]
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                c=center_colors, s=200, marker="X", edgecolor="k")
    """

    ax.set_xlabel(feat_to_plot[0])
    ax.set_ylabel(feat_to_plot[1])
    ax.set_zlabel(feat_to_plot[2])
    ax.set_title(f"K-Means ({n_clusters} clusters)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{ii}__k-means_clustering_3d.png"),
                dpi=300, bbox_inches="tight")
    plt.show()

  else:
      # -------- 2D PLOT (your original) --------
      plt.figure(figsize=(6,6))
      point_colors = [cluster_colors[int(c)] for c in labels]
      plt.scatter(X_to_plot[:,0], X_to_plot[:,1], c=point_colors, s=25)
      center_colors = [cluster_colors[i] for i in range(n_clusters)]
      plt.scatter(centers[:,0], centers[:,1], c=center_colors,
                  s=200, marker="X", edgecolor="k")
      plt.xlabel(feat_for_clustering[0]); plt.ylabel(feat_for_clustering[1])
      plt.tight_layout()
      plt.savefig(os.path.join(PLOTS_DIR, f"{ii}__k-means_clustering.png"),
                  dpi=300, bbox_inches="tight")

  # ---- Distance to cluster centroid (works for 2D or 3D+) ----
  # centers[df["km_label"]] uses advanced indexing; make sure labels are int
  df["dist_from_centroids"] = np.linalg.norm(
      df[feat_for_clustering].to_numpy() - centers[labels], axis=1
  )