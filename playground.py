from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from decouple import config
from backend import ResearchQueryBackend
import pandas as pd

cohere_llm = ChatCohere(cohere_api_key=config("COHERE_API_KEY"), temperature=0.6)

prompt_template = """
You are a knowledgeable assistant with access to a database of research papers. 
The user asked the following question: "{query}"

Based on the query, here are the top results from the research paper database:

{search_results}

Please provide a concise and helpful response to the user based on these results.
"""

template = PromptTemplate(input_variables=["query", "search_results"], template=prompt_template)

chain = template | cohere_llm


class ResearchQueryWithLLM(ResearchQueryBackend):
    def query_with_llm(self, user_query, use_pinecone=False):
        if use_pinecone:
            search_results = self.query_pinecone_index(user_query)
        else:
            search_results = self.query_annoy_index(user_query)

        if isinstance(search_results, pd.DataFrame):
            formatted_results = ""
            for _, row in search_results.iterrows():
                abstract = row.get('Abstract', 'N/A')
                authors = row.get('Authors', 'N/A')
                year = row.get('Publication Year', 'N/A')
                formatted_results += f"Abstract: {abstract}\nAuthors: {authors}\nYear: {year}\n\n"
        else:
            formatted_results = "No results found."

        response = chain.invoke({"query": user_query, "search_results": formatted_results})
        
        return response

research_llm = ResearchQueryWithLLM()

research_llm.load_data('data/data.csv')

research_llm.create_embeddings()
research_llm.create_annoy_index()
# research_llm.create_pinecone_index('mh')
# research_llm.upsert_to_pinecone()

user_query = "What happens during a surgery?"
response = research_llm.query_with_llm(user_query, use_pinecone=False)

print(response)