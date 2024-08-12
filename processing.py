import streamlit as st
import cohere
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from decouple import config
from pinecone import Pinecone, ServerlessSpec

# Initialize Cohere and Pinecone clients
@st.cache_resource
def init_clients():
    cohere_api_key = config("COHERE_API_KEY")
    pinecone_api_key = config("PINECONE_API_KEY")
    co = cohere.Client(cohere_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    return co, pc

# Load and preprocess data
@st.cache_data
def load_data(file_path):
    raw_df = pd.read_csv(file_path, index_col=0)
    df = raw_df.drop(columns=['Summary'])
    df.dropna(inplace=True)
    return df

# Create Annoy index
@st.cache_resource
def create_annoy_index(embeds, n_trees=100):
    search_index = AnnoyIndex(embeds.shape[1], 'angular')
    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])
    search_index.build(n_trees)
    return search_index

# Query Annoy index
def query_annoy_index(search_index, query, df, co):
    query_embed = co.embed(texts=[query], model='multilingual-22-12').embeddings
    similar_item_ids, _ = search_index.get_nns_by_vector(query_embed[0], 5, include_distances=True)
    
    if len(similar_item_ids) >= 2:
        results = pd.DataFrame(data={
            'Abstract': df.iloc[similar_item_ids[0]]['Abstract'],
            'Authors': df.iloc[similar_item_ids[0]]['Authors'],
            'Publication Year': df.iloc[similar_item_ids[0]]['Publication Year'],
        }, index=[0])
        return results
    else:
        return pd.DataFrame()

# Create Pinecone index
@st.cache_resource
def create_pinecone_index(_pc, index_name, dimension):
    if index_name not in _pc.list_indexes().names():
        _pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return _pc.Index(index_name)

# Upsert to Pinecone
@st.cache_resource
def upsert_to_pinecone(_index, df, embeds, batch_size=128):
    ids = [str(i) for i in range(len(df))]
    meta = [
        {'Abstract': abstract, 'Authors': authors, 'Publication Year': publication_year}
        for abstract, authors, publication_year in zip(df['Abstract'], df['Authors'], df['Publication Year'])
    ]
    to_upsert = list(zip(ids, embeds, meta))
    
    for i in range(0, len(df), batch_size):
        i_end = min(i + batch_size, len(df))
        _index.upsert(vectors=to_upsert[i:i_end])

# Query Pinecone index
def query_pinecone_index(index, query, co):
    xq = co.embed(texts=[query], model='multilingual-22-12', truncate='NONE').embeddings
    res = index.query(vector=xq, top_k=5, include_metadata=True)
    return res['matches']

# Streamlit app
def main():
    st.title("Research Paper Query App")

    # Initialize clients
    co, pc = init_clients()

    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df = load_data("data/data.csv")
    st.success("Data loaded successfully!")

    # Create embeddings
    with st.spinner("Creating embeddings..."):
        texts = df["Abstract"].tolist()
        embeds = np.array(co.embed(texts=texts, model='multilingual-22-12').embeddings)
    st.success("Embeddings created successfully!")

    # Create Annoy index
    with st.spinner("Creating Annoy index..."):
        search_index = create_annoy_index(embeds)
    st.success("Annoy index created successfully!")

    # Create Pinecone index
    with st.spinner("Creating Pinecone index..."):
        index_name = 'mh'
        dimension = embeds.shape[1]
        index = create_pinecone_index(pc, index_name, dimension)
    st.success("Pinecone index created successfully!")

    # Upsert data to Pinecone
    with st.spinner("Upserting data to Pinecone..."):
        upsert_to_pinecone(index, df, embeds)
    st.success("Data upserted to Pinecone successfully!")

    # User input
    query = st.text_input("Enter your research query:")
    search_button = st.button("Search")

    if search_button and query:
        # Query Annoy index
        with st.spinner("Searching Annoy index..."):
            annoy_results = query_annoy_index(search_index, query, df, co)
        
        # Display Annoy results
        st.subheader("Annoy Search Results")
        if not annoy_results.empty:
            st.dataframe(annoy_results)
        else:
            st.warning("No results found in Annoy index.")

        # Query Pinecone index
        with st.spinner("Searching Pinecone index..."):
            pinecone_results = query_pinecone_index(index, query, co)
        
        # Display Pinecone results
        st.subheader("Pinecone Search Results")
        if pinecone_results:
            for i, match in enumerate(pinecone_results, 1):
                st.write(f"Result {i}:")
                st.write(f"Score: {match['score']:.2f}")
                st.write(f"Abstract: {match['metadata']['Abstract']}")
                st.write(f"Authors: {match['metadata']['Authors']}")
                st.write(f"Publication Year: {match['metadata']['Publication Year']}")
                st.write("---")
        else:
            st.warning("No results found in Pinecone index.")

if __name__ == "__main__":
    main()