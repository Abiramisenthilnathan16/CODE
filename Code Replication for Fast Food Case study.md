import pandas as pd

# Load data
mcdonalds = pd.read_csv("https://homepage.boku.ac.at/leisch/MSA/datasets/mcdonalds.csv")

# Print column names
print(mcdonalds.columns)

# Print dimensions
print(mcdonalds.shape)

# Print first 3 rows
print(mcdonalds.head(3))
# Select columns 1 to 11 and convert "Yes" to 1 and "No" to 0
MD_x = mcdonalds.iloc[:, 0:11].apply(lambda x: (x == "Yes") + 0)

# Compute column means
column_means = round(MD_x.mean(), 2)
print(column_means)
from sklearn.decomposition import PCA


# Perform PCA
pca = PCA()
MD_pca = pca.fit(MD_x)

# Print summary
print("Importance of components:")
print(pd.DataFrame({
    'Standard deviation': MD_pca.explained_variance_,
    'Proportion of Variance': MD_pca.explained_variance_ratio_,
    'Cumulative Proportion': np.cumsum(MD_pca.explained_variance_ratio_)
}))
import numpy as np

# Print standard deviations
print("Standard deviations (1, .., p=11):")
print(np.round(MD_pca.singular_values_, 1))

# Print rotation matrix (loadings)
print("Rotation (n x k) = (11 x 11):")
print(pd.DataFrame(np.round(MD_pca.components_.T, 3), index=MD_x.columns))
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming MD_pca is already computed using PCA

# Transform data to PCA space
MD_pca_transformed = MD_pca.transform(MD_x)

# Plot transformed data
plt.scatter(MD_pca_transformed[:, 0], MD_pca_transformed[:, 1], color='grey')

# Add labels and title
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')

# Add projection axes
for i, (pcx, pcy) in enumerate(zip(MD_pca.components_[0], MD_pca.components_[1])):
    plt.arrow(0, 0, pcx, pcy, color='red', alpha=0.5)
    plt.text(pcx, pcy, MD_x.columns[i], color='red')

# Show plot
plt.show()
from sklearn.cluster import KMeans
import numpy as np

# Set random seed
np.random.seed(1234)

# Perform K-means clustering
cluster_results = {}
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    cluster_results[k] = kmeans.labels_

# Relabel clusters
def relabel_clusters(labels):
    unique_labels = np.unique(labels)
    relabeled = labels.copy()
    for i, label in enumerate(unique_labels):
        relabeled[labels == label] = i
    return relabeled

for k, labels in cluster_results.items():
    cluster_results[k] = relabel_clusters(labels)
# Calculate within-cluster sum of squares
wcss = []
for k, labels in cluster_results.items():
    centroids = [np.mean(MD_x[labels == i], axis=0) for i in range(k)]
    dist_sum = sum(np.sum((MD_x[labels == i] - centroids[i])**2) for i in range(k))
    wcss.append(dist_sum)

# Print within-cluster sum of squares
print("Within-Cluster Sum of Squares:")
for k, dist_sum in zip(range(2, 9), wcss):
    print(f"Number of Segments: {k}, WCSS: {dist_sum}")

# Plot the within-cluster sum of squares
plt.figure(figsize=(8, 6))
plt.plot(range(2, 9), wcss, marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()
from sklearn.utils import resample

# Set random seed
np.random.seed(1234)

# Define the number of bootstrap repetitions
n_bootstrap = 100

# Perform bootstrap clustering
bootstrap_results = {}
for k in range(2, 9):
    bootstrap_results[k] = []
    for _ in range(n_bootstrap):
        # Bootstrap resampling
        bootstrap_sample = resample(MD_x)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
        kmeans.fit(bootstrap_sample)
        
        # Store clustering labels
        bootstrap_results[k].append(kmeans.labels_)

# Convert results to numpy arrays
for k in range(2, 9):
    bootstrap_results[k] = np.array(bootstrap_results[k])
from sklearn.metrics import adjusted_rand_score

# Calculate adjusted Rand index for each bootstrap sample
ari_scores = {}
for k, labels_matrix in bootstrap_results.items():
    ari_scores[k] = [adjusted_rand_score(labels_matrix[i], labels_matrix[j])
                     for i in range(n_bootstrap)
                     for j in range(i + 1, n_bootstrap)]

# Plot the mean adjusted Rand index
mean_ari = {k: np.mean(scores) for k, scores in ari_scores.items()}
plt.figure(figsize=(8, 6))
plt.plot(list(mean_ari.keys()), list(mean_ari.values()), marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Adjusted Rand Index for Bootstrap Clustering')
plt.grid(True)
plt.show()

# Selecting cluster memberships for a specific number of clusters, let's say 4
cluster_memberships = cluster_results[4]

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(cluster_memberships, bins=np.arange(5), align='left', edgecolor='black')
plt.xlabel('Cluster Membership')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Memberships')
plt.xlim(0, 4)
plt.xticks(np.arange(4))
plt.grid(axis='y')
plt.show()
MD_k4 = cluster_results[4]
from sklearn.neighbors import LocalOutlierFactor


# Initialize LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')

# Fit LOF to each cluster separately
MD_r4 = {}
for cluster_id in range(4):
    cluster_indices = np.where(MD_k4 == cluster_id)[0]
    lof.fit_predict(MD_x.iloc[cluster_indices])  # Using iloc for DataFrame indexing
    MD_r4[cluster_id] = lof.negative_outlier_factor_
import matplotlib.pyplot as plt


# Print segment stability values
for cluster_id, stability_values in MD_r4.items():
    print(f'Cluster {cluster_id} stability values:')
    print(stability_values)

# Plot segment stability
plt.figure(figsize=(8, 6))
for cluster_id, stability_values in MD_r4.items():
    plt.plot(stability_values, label=f'Cluster {cluster_id}')
plt.xlabel('Segment Number')
plt.ylabel('Segment Stability')
plt.title('Segment Stability for Each Cluster')
plt.ylim(0, 1)  # Set y-axis limits
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
from sklearn.mixture import GaussianMixture

# Set random seed
np.random.seed(1234)

# Assuming MD_x is your data

# Fit Gaussian Mixture Model with 2 to 8 components
bic_values = []
for k in range(2, 9):
    bic = []
    for _ in range(10):  # Repeat 10 times
        gmm = GaussianMixture(n_components=k, random_state=1234)
        gmm.fit(MD_x)
        bic.append(gmm.bic(MD_x))
    bic_values.append(bic)

# Find the number of components with the lowest BIC
best_k_index = np.argmin(np.mean(bic_values, axis=1)) + 2  # Add 2 to get the actual number of components
print("Best number of components:", best_k_index)

# Note: You can access other information such as log-likelihood, AIC, etc. by modifying the code accordingly.
import matplotlib.pyplot as plt


# Plot AIC, BIC, and ICL
plt.figure(figsize=(10, 6))
for i, criterion in enumerate(['AIC', 'BIC', 'ICL']):
    plt.plot(range(2, 9), [np.mean(bic_vals) for bic_vals in bic_values], label=criterion)
plt.xlabel('Number of Components')
plt.ylabel('Value of Information Criterion')
plt.title('Model Selection Criteria')
plt.legend()
plt.grid(True)
plt.show()

# Convert the 'Like' variable to numeric
mcdonalds['Like'] = pd.to_numeric(mcdonalds['Like'], errors='coerce')

# Reverse the counts of the Like variable
reversed_like_counts = mcdonalds['Like'].value_counts().sort_index(ascending=False)

# Print the reversed counts
print("Reversed counts of Like variable:")
print(reversed_like_counts)

# Create a new variable Like.n by subtracting each value from 6
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Print the counts of the new variable Like.n
print("Counts of the new variable Like.n:")
print(mcdonalds['Like.n'].value_counts().sort_index())
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Assuming MD.x is the data matrix
MD_x_transposed = np.transpose(MD_x)

# Perform hierarchical clustering
Z = linkage(MD_x_transposed, method='average', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
# Define hierarchical clustering order (assuming it's a list of indices)
MD_vclust_order = list(range(len(MD_x_transposed)))

# Get unique cluster assignments
unique_clusters = np.unique(MD_k4)

# Plot bar chart with shading
plt.figure(figsize=(10, 6))
for i, order_index in enumerate(reversed(MD_vclust_order)):
    cluster_assignments = MD_k4[order_index]
    cluster_counts = [np.sum(cluster_assignments == cluster) for cluster in unique_clusters]
    plt.bar(unique_clusters, cluster_counts, color=plt.cm.viridis(i / len(MD_vclust_order)), alpha=0.7, label=f'Cluster {i+1}')

plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Bar Chart of Cluster Assignments with Shading based on Hierarchical Clustering Order')
plt.legend()
plt.show()

from statsmodels.graphics.mosaicplot import mosaic

# Create a DataFrame with cluster assignments and Like variable
data = pd.DataFrame({'Cluster': MD_k4, 'Like': mcdonalds['Like']})

# Create the mosaic plot
plt.figure(figsize=(10, 8))
mosaic(data, ['Cluster', 'Like'], title='Mosaic Plot of Cluster vs. Like')
plt.xlabel('Segment Number')
plt.show()

from statsmodels.graphics.mosaicplot import mosaic
# Create a DataFrame with cluster assignments and Gender variable
data = pd.DataFrame({'Cluster': MD_k4, 'Gender': mcdonalds['Gender']})

# Create the mosaic plot
plt.figure(figsize=(10, 8))
mosaic(data, ['Cluster', 'Gender'], title='Mosaic Plot of Cluster vs. Gender', gap=0.02)
plt.show()
import pandas as pd

# Assuming k4 represents cluster assignments and is an integer array or list
k4_series = pd.Series(k4)

# Check unique values in k4_series
print(k4_series.unique())
import pandas as pd

# Assuming k4 represents cluster assignments and is an integer array or list
k4_series = pd.Series(k4)

# Check unique values in k4_series
print(k4_series.unique())
# Assuming MD_k4 contains cluster assignments and mcdonalds is a DataFrame containing the data

# Convert Gender to binary (1 for Female, 0 for Male)
mcdonalds['IsFemale'] = (mcdonalds['Gender'] == 'Female').astype(int)

# Compute the proportion of females in each cluster
female = mcdonalds.groupby(MD_k4)['IsFemale'].mean()

# Print the proportion of females in each cluster
print(female)
