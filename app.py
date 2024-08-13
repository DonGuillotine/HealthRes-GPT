import streamlit as st
from pathlib import Path
from backend import ResearchQueryBackend

@st.cache_resource
def get_backend():
    return ResearchQueryBackend()

def main():
    st.title("HealthRes-GPT")

    backend = get_backend()

    # Load data
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading and preprocessing data..."):
            shape = backend.load_data("data/data.csv")
        st.success(f"Data loaded successfully! Shape: {shape}")
        st.session_state.data_loaded = True

    # Create embeddings
    if 'embeddings_created' not in st.session_state:
        with st.spinner("Creating embeddings..."):
            shape = backend.create_embeddings()
        st.success(f"Embeddings created successfully! Shape: {shape}")
        st.session_state.embeddings_created = True

    # Create Annoy index
    if 'annoy_index_created' not in st.session_state:
        with st.spinner("Creating Annoy index..."):
            backend.create_annoy_index()
        st.success("Annoy index created successfully!")
        st.session_state.annoy_index_created = True

    # Create Pinecone index
    if 'pinecone_index_created' not in st.session_state:
        with st.spinner("Creating Pinecone index..."):
            backend.create_pinecone_index('mh')
        st.success("Pinecone index created successfully!")
        st.session_state.pinecone_index_created = True

    # Upsert data to Pinecone
    if 'data_upserted' not in st.session_state:
        with st.spinner("Upserting data to Pinecone..."):
            backend.upsert_to_pinecone()
        st.success("Data upserted to Pinecone successfully!")
        st.session_state.data_upserted = True

    # User input
    query = st.text_input("Enter your research query:")
    search_button = st.button("Search")

    if search_button and query:
        # Query Annoy index
        with st.spinner("Searching Annoy index..."):
            annoy_results = backend.query_annoy_index(query)
        
        # Display Annoy results
        st.subheader("Annoy Search Results")
        if not annoy_results.empty:
            st.dataframe(annoy_results)
        else:
            st.warning("No results found in Annoy index.")

        # Query Pinecone index
        with st.spinner("Searching Pinecone index..."):
            pinecone_results = backend.query_pinecone_index(query)
        
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
