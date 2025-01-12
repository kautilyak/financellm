# src/knowledge_base/vector_store.py
from typing import List, Dict
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os


class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model
        )

        # Initialize or load existing vector store
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the vector store"""
        return Chroma(
            persist_directory=self.config.vector_store_path,
            embedding_function=self.embeddings,
            collection_name="stock_data",
        )

    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store"""
        try:
            self.vector_store.add_documents(documents) # Save to disk
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        Returns k most similar documents.
        """
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []