import streamlit as st
from pathlib import Path
from playground import ResearchQueryWithLLM

@st.cache_resource
def get_backend():
    return ResearchQueryWithLLM()

def main():
    st.title("HealthRes-GPT")

    backend = get_backend()

    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading and preprocessing data..."):
            shape = backend.load_data("data/data.csv")
        st.success(f"Data loaded successfully! Shape: {shape}")
        st.session_state.data_loaded = True

    if 'embeddings_created' not in st.session_state:
        with st.spinner("Creating embeddings..."):
            shape = backend.create_embeddings()
        st.success(f"Embeddings created successfully! Shape: {shape}")
        st.session_state.embeddings_created = True

    if 'annoy_index_created' not in st.session_state:
        with st.spinner("Creating Annoy index..."):
            backend.create_annoy_index()
        st.success("Annoy index created successfully!")
        st.session_state.annoy_index_created = True

    # Create Pinecone index
    # if 'pinecone_index_created' not in st.session_state:
    #     with st.spinner("Creating Pinecone index..."):
    #         backend.create_pinecone_index('mh')
    #     st.success("Pinecone index created successfully!")
    #     st.session_state.pinecone_index_created = True

    # Upsert data to Pinecone
    # if 'data_upserted' not in st.session_state:
    #     with st.spinner("Upserting data to Pinecone..."):
    #         backend.upsert_to_pinecone()
    #     st.success("Data upserted to Pinecone successfully!")
    #     st.session_state.data_upserted = True

    index_option = st.radio(
        "Choose the index to use for querying:",
        ('Annoy', 'Pinecone')
    )
    use_pinecone = index_option == 'Pinecone'

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    user_query = st.text_input("Ask your question:")
    if st.button("Send"):
        if user_query:
            with st.spinner("Generating response..."):
                response = backend.query_with_llm(user_query, use_pinecone=use_pinecone)

            st.session_state.conversation_history.append({"role": "user", "content": user_query})
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

    st.subheader("Chatbot Conversation")
    for chat in st.session_state.conversation_history:
        if chat['role'] == 'user':
            st.write(f"**You:** {chat['content']}")
        else:
            st.write(f"**HealthRes-GPT:** {chat['content']}")

if __name__ == "__main__":
    main()
