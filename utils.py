import pdfplumber
import openai

def store_vector(embedding, metadata, index):
    index.upsert([(metadata['id'], embedding, metadata)])

def get_embedding(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def query_pinecone(query_text, api_key, index):
    query_embedding = get_embedding(query_text, api_key)
    result = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
    return result

def extract_text_from_file(file):
    if file.type == "application/pdf":
        # Use a PDF parser to extract text
        # Example: use PyPDF2 or pdfplumber
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    else:
        # Assume it's a text file
        return file.getvalue().decode("utf-8")