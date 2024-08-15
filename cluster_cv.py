import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the dataset
file_path = 'dataset/restaruant.csv'
df = pd.read_csv(file_path)

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for the text data
embeddings = model.encode(df['text'].tolist())

# Determine the best number of clusters using the Elbow Method
inertia = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    inertia.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# # Plot the Elbow Method graph
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.plot(K, inertia, 'bo-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.grid(True)

# # Plot the Silhouette Scores
# plt.subplot(1, 2, 2)
# plt.plot(K, silhouette_scores, 'bo-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Scores for Optimal k')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# Determine the best k (highest silhouette score)
best_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f'The optimal number of clusters is: {best_k}')

# Perform KMeans clustering with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=best_k, random_state=42)
df['cluster_optimal'] = kmeans_optimal.fit_predict(embeddings)

# Visualize the clustering using PCA
reduced_data_bert = PCA(n_components=2).fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_data_bert[:, 0], reduced_data_bert[:, 1], c=df['cluster_optimal'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title(f'Text Clustering Visualization using BERT Embeddings with k={best_k}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Save the clustered dataset
df.to_csv('dataset/clustered_npr.csv', index=False)
