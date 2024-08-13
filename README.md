# HealthRes-GPT
![image](https://github.com/user-attachments/assets/d84b0312-2aa1-4896-9890-fe2b16e6c422)

## Description

This Research Paper Query App is a Streamlit-based web application that allows users to search through a database of research papers using both Annoy and Pinecone indexing methods. It provides a user-friendly interface for querying research papers and viewing detailed statistics about the index.

## Features

- Load and preprocess research paper data
- Create embeddings for research paper abstracts
- Build and query Annoy index for fast approximate nearest neighbor search
- Create and query Pinecone index for scalable vector search
- User-friendly interface for entering queries
- Display search results from both Annoy and Pinecone indexes
- View Pinecone index statistics

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/DonGuillotine/ai-powered-personal-health-dashboard.git
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the root directory of the project.
2. Add your API keys to the `.env` file:
   ```bash
   COHERE_API_KEY=your_cohere_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## Usage

1. Ensure your research paper data is in a CSV file named `data.csv` in the project directory.

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and go to the URL displayed in the terminal (`http://localhost:8501`).

4. Use the text input to enter your research query and click the "Search" button to view results.

## Project Structure

- `app.py`: Contains the Streamlit user interface code.
- `backend.py`: Contains the backend logic for data processing, index creation, and querying.
- `requirements.txt`: Lists all the Python dependencies for the project.
- `.env`: (Not in repository) Contains your API keys.
- `data.csv`: (Not in repository) Your research paper dataset.

## Dependencies

- streamlit
- cohere
- pandas
- numpy
- annoy
- pinecone-client
- python-decouple

## Contributing

Contributions to improve the app are welcome. Please feel free to submit a Pull Request.

## Acknowledgments

- [Muhammad Inaamullah](https://github.com/m-inaam)
- Cohere for providing the embedding model
- Pinecone for the vector database service
- Annoy for the approximate nearest neighbors algorithm

## Future plans
Make an actual dashboard haha
