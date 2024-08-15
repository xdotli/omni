import pandas as pd
import click
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to perform clustering
def perform_clustering(df, n_clusters):
    embeddings = model.encode(df['text'].tolist())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)
    return df, embeddings

# Function to determine the number of clusters
def determine_clusters(num_rows):
    return min(10, max(1, num_rows // 50))

# Function to sample rows from clusters
def sample_clusters(df, n_clusters):
    samples = []
    for cluster_id in range(n_clusters):
        cluster_df = df[df['cluster'] == cluster_id]
        samples.append(cluster_df.sample(min(10, len(cluster_df))))
    return pd.concat(samples)

# Function to simulate LLM category refinement
def refine_categories(samples):
    # Simulated refinement using an LLM (this could be replaced with an actual LLM API call)
    # For now, we return a list of unique categories extracted from the samples
    return list(set(samples['cluster'].apply(lambda x: f'Category-{x}')))

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output_file', default='output.csv', help='Output CSV file name')
def main(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Determine the number of clusters
    n_clusters = determine_clusters(len(df))

    # Perform clustering
    df, embeddings = perform_clustering(df, n_clusters)

    # Sample rows from each cluster
    samples = sample_clusters(df, n_clusters)

    # Refine categories using a simulated LLM
    refined_categories = refine_categories(samples)

    # Print the refined categories
    print("Refined Categories:", refined_categories)

    # Map the clusters back to refined categories
    df['category'] = df['cluster'].map(lambda x: refined_categories[x % len(refined_categories)])

    # Save the final dataset with categories to a CSV file
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
