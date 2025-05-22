from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self):
        # Use local sentence transformer
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
    
    def add_documents(self, documents):
        ids = [f"{doc['metadata']['doc_id']}_{doc['metadata']['chunk_id']}" for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def query(self, text, k=5):
        results = self.collection.query(
            query_texts=[text],
            n_results=k
        )
        return results