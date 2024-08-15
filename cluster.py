import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset/npr.csv'
df = pd.read_csv(file_path)

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for the text data
embeddings = model.encode(df['text'].tolist())

# Perform KMeans clustering on the embeddings
kmeans = KMeans(n_clusters=20, random_state=42)
df['cluster_bert'] = kmeans.fit_predict(embeddings)

# Visualize the clustering using PCA
reduced_data_bert = PCA(n_components=2).fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_data_bert[:, 0], reduced_data_bert[:, 1], c=df['cluster_bert'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Text Clustering Visualization using BERT Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Save the clustered dataset
df.to_csv('dataset/clustered_npr.csv', index=False)
