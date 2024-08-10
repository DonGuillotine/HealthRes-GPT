import streamlit as st
from openai import OpenAI
import pinecone
from utils import extract_text_from_file, query_pinecone, get_embedding, store_vector

# Show title and description.
st.title("AI Powered Personal Health Dashboard")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.text_input("Pinecone API Key", type="password")

if not openai_api_key or not pinecone_api_key:
    st.info("Please add your OpenAI and Pinecone API keys to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
    index_name = "document-embeddings"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)  
    index = pinecone.Index(index_name)

    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            text = extract_text_from_file(file)
            embedding = get_embedding(text, openai_api_key)
            metadata = {"id": file.name, "text": text}
            store_vector(embedding, metadata, index)
            st.success(f"Stored {file.name} in Pinecone.")

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = query_pinecone(prompt, openai_api_key, index)
        for match in result['matches']:
            st.write(f"**{match['metadata']['id']}**: {match['metadata']['text'][:200]}...")

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})