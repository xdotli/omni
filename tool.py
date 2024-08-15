import os
from openai import OpenAI
import pandas as pd
import click
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import random
import json

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
MODEL = "gpt-4o-2024-08-06"
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

# Function to refine categories using OpenAI structured outputs
def refine_categories(samples):
    prompt = '''
    You are an AI assistant that refines text clustering by assigning categories.
    The input is a set of text clusters, and your task is to provide a category for each cluster.
    These texts have been clustered together based on their content using BERT embeddings and KMeans clustering.
    '''

    # Sample JSON schema
    json_schema = {
        "name": "text_category",
        'schema': {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["categories"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Prepare the input for the LLM
    # clusters = samples.groupby('cluster')['text'].apply(lambda x: "\n".join(x))
    clusters = samples.groupby('cluster')['text'].apply(lambda x: "\n".join(x)).to_dict()
    clusters_json = json.dumps(clusters)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": clusters_json}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": json_schema,
        }
    )

    return response.choices[0].message.content

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

    # Refine categories using OpenAI's structured output
    refined_categories = refine_categories(samples)

    refined_categories = json.loads(refined_categories)

    refined_categories = refined_categories['categories']

    # Print the refined categories
    print("Refined Categories:", refined_categories)

    # Map the clusters back to refined categories
    df['category'] = df['cluster'].map(lambda x: refined_categories[x % len(refined_categories)])

    # Save the final dataset with categories to a CSV file
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
