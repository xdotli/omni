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
def refine_categories(samples, user_prompt):
    prompt = f'''
    You are an AI assistant that refines text clustering by assigning categories.
    The input is a set of text clusters, and your task is to provide a category based on the clusters.
    The number of categories doesn't have to be the same as the number of clusters. Make sensible categories.
    These texts have been clustered together based on their content using BERT embeddings and KMeans clustering.
    User input: {user_prompt}
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
    clusters = samples['text'].apply(lambda x: "\n".join(x)).to_dict()
    # drop the cluster column:
    clusters_json = json.dumps(clusters)
    print(clusters_json)

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

# Function to generate categories for each chunk
def generate_categories_for_chunks(df, refined_categories, user_prompt):
    chunk_size = 20
    df['category'] = ''

    json_schema = {
        "name": "row",
        'schema': {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "category": {"type": "string"}
                        },
                        "required": ["id", "category"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["rows"],
            "additionalProperties": False
        },
        "strict": True
    }


    for i in range(0, len(df), chunk_size):  # Iterate through chunks of data
        chunk = df.iloc[i:i + chunk_size]
        chunk_prompt = f'''
        You are an AI assistant that generates categories for text data.
        Here is a chunk of data with its tentative clusters and a refined category set provided by the user.
        Use the refined categories and the user input to generate a suitable category for each row.
        Refined Categories: {', '.join(refined_categories)}
        User input: {user_prompt}
        '''


        chunk_input = chunk.to_json(orient='records')

        print(chunk_input)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": chunk_prompt},
                {"role": "user", "content": chunk_input}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            }
        )

        chunk_categories = json.loads(response.choices[0].message.content)
        print(chunk_categories)

        # Assuming the response is in the format of a list of categories corresponding to each row
        for item in chunk_categories['rows']:
            id_value = int(item['id'])  # Convert id to integer
            df.loc[df['id'] == id_value, 'category'] = item['category']
            # print(f"ID: {id_value}, Category: {item['category']}")
            # print(df.loc[df['id'] == id_value])

    return df

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output_file', default='output.csv', help='Output CSV file name')
@click.option('--user_input', prompt='Please provide additional input for the LLM:', help='User input to be passed to the LLM')
def main(input_file, output_file, user_input):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Determine the number of clusters
    n_clusters = determine_clusters(len(df))

    # Perform clustering
    df, embeddings = perform_clustering(df, n_clusters)

    # Sample rows from each cluster
    samples = sample_clusters(df, n_clusters)
    # drop the cluster column:
    samples = samples.drop('cluster', axis=1)

    # Refine categories using OpenAI's structured output
    refined_categories = refine_categories(samples, user_input)
    refined_categories = json.loads(refined_categories)['categories']

    # Generate categories for each chunk
    df = generate_categories_for_chunks(df, refined_categories, user_input)

    # Save the final dataset with categories to a CSV file
    df.to_csv(output_file, index=False)

    print(f"Categories have been generated and saved to {output_file}")

if __name__ == '__main__':
    main()
