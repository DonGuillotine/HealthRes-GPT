import cohere
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from decouple import config
from pinecone import Pinecone, ServerlessSpec

class ResearchQueryBackend:
    def __init__(self):
        self.co = cohere.Client(config("COHERE_API_KEY"))
        self.pc = Pinecone(api_key=config("PINECONE_API_KEY"))
        self.df = None
        self.embeds = None
        self.annoy_index = None
        self.pinecone_index = None

    def load_data(self, file_path):
        raw_df = pd.read_csv(file_path, index_col=0)
        self.df = raw_df.drop(columns=['Summary'])
        self.df.dropna(inplace=True)
        return self.df.shape

    def create_embeddings(self):
        texts = self.df["Abstract"].tolist()
        self.embeds = np.array(self.co.embed(texts=texts, model='multilingual-22-12').embeddings)
        return self.embeds.shape

    def create_annoy_index(self, n_trees=100):
        self.annoy_index = AnnoyIndex(self.embeds.shape[1], 'angular')
        for i in range(len(self.embeds)):
            self.annoy_index.add_item(i, self.embeds[i])
        self.annoy_index.build(n_trees)
        return True

    def query_annoy_index(self, query):
        query_embed = self.co.embed(texts=[query], model='multilingual-22-12').embeddings
        similar_item_ids, _ = self.annoy_index.get_nns_by_vector(query_embed[0], 5, include_distances=True)
        
        if len(similar_item_ids) >= 2:
            results = pd.DataFrame(data={
                'Abstract': self.df.iloc[similar_item_ids[0]]['Abstract'],
                'Authors': self.df.iloc[similar_item_ids[0]]['Authors'],
                'Publication Year': self.df.iloc[similar_item_ids[0]]['Publication Year'],
            }, index=[0])
            return results
        else:
            return pd.DataFrame()

    def create_pinecone_index(self, index_name):
        dimension = self.embeds.shape[1]
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.pinecone_index = self.pc.Index(index_name)
        return True

    def upsert_to_pinecone(self, batch_size=128):
        ids = [str(i) for i in range(len(self.df))]
        meta = [
            {'Abstract': abstract, 'Authors': authors, 'Publication Year': publication_year}
            for abstract, authors, publication_year in zip(self.df['Abstract'], self.df['Authors'], self.df['Publication Year'])
        ]
        to_upsert = list(zip(ids, self.embeds, meta))
        
        for i in range(0, len(self.df), batch_size):
            i_end = min(i + batch_size, len(self.df))
            self.pinecone_index.upsert(vectors=to_upsert[i:i_end])
        return True

    def query_pinecone_index(self, query):
        xq = self.co.embed(texts=[query], model='multilingual-22-12', truncate='NONE').embeddings
        res = self.pinecone_index.query(vector=xq, top_k=5, include_metadata=True)
        return res['matches']