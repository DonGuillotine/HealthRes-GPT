import cohere
import os
import random
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from decouple import config
from pinecone import Pinecone, ServerlessSpec

def load_data(file_path):
    """Load and preprocess the data."""
    raw_df = pd.read_csv(file_path, index_col=0)
    print(f"Original data shape: {raw_df.shape}")
    
    df = raw_df.drop(columns=['Summary'])
    df.dropna(inplace=True)
    print(f"Processed data shape: {df.shape}")
    
    return df

def print_sample_documents(texts, n=5):
    """Print n sample documents."""
    random.seed(100)
    for item in random.sample(texts, n):
        print(item)

def create_annoy_index(embeds, n_trees=100):
    """Create and save Annoy index."""
    search_index = AnnoyIndex(embeds.shape[1], 'angular')
    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])
    search_index.build(n_trees)
    search_index.save('data_index.ann')
    return search_index

def query_annoy_index(search_index, query, df, co):
    """Query the Annoy index."""
    query_embed = co.embed(texts=[query], model='multilingual-22-12').embeddings
    similar_item_ids, _ = search_index.get_nns_by_vector(query_embed[0], 5, include_distances=True)
    
    if len(similar_item_ids) >= 2:
        results = pd.DataFrame(data={
            'Abstract': df.iloc[similar_item_ids[0]]['Abstract'],
            'Authors': df.iloc[similar_item_ids[0]]['Authors'],
            'Publication Year': df.iloc[similar_item_ids[0]]['Publication Year'],
        }, index=[0])
        print(f"Query: '{query}'\nNearest neighbors:")
        print(results)
    else:
        print("Not enough similar items found for query:", query)

def create_pinecone_index(pc, index_name, dimension):
    """Create Pinecone index."""
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    return pc.Index(index_name)

def upsert_to_pinecone(index, df, embeds, batch_size=128):
    """Upsert data to Pinecone index."""
    ids = [str(i) for i in range(len(df))]
    meta = [
        {'Abstract': abstract, 'Authors': authors, 'Publication Year': publication_year}
        for abstract, authors, publication_year in zip(df['Abstract'], df['Authors'], df['Publication Year'])
    ]
    to_upsert = list(zip(ids, embeds, meta))
    
    for i in range(0, len(df), batch_size):
        i_end = min(i + batch_size, len(df))
        index.upsert(vectors=to_upsert[i:i_end])

def query_pinecone_index(index, query, co):
    """Query the Pinecone index."""
    xq = co.embed(texts=[query], model='multilingual-22-12', truncate='NONE').embeddings
    res = index.query(vector=xq, top_k=10, include_metadata=True)
    
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata']['Abstract']}")
        print(f"{match['score']:.2f}: {match['metadata']['Authors']}")
        print(f"{match['score']:.2f}: {match['metadata']['Publication Year']}")
        print()

def main():
    api_key = config("COHERE_API_KEY")
    co = cohere.Client(api_key)

    df = load_data("data/data.csv")

    texts = df["Abstract"].tolist()
    embeds = np.array(co.embed(texts=texts, model='multilingual-22-12').embeddings)

    search_index = create_annoy_index(embeds)
    query_annoy_index(search_index, "What is the best menstrual cycle?", df, co)

    pc = Pinecone(api_key=config("PINECONE_API_KEY"))
    index_name = 'mh'
    dimension = embeds.shape[1]

    index = create_pinecone_index(pc, index_name, dimension)

    upsert_to_pinecone(index, df, embeds)

    print("Pinecone index statistics:")
    print(index.describe_index_stats())
    
    query_pinecone_index(index, "What is the best menstrual cycle?", co)

if __name__ == "__main__":
    main()