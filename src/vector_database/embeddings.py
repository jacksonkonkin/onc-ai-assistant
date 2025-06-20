"""
Embedding management and generation
Teams: Data team + LLM team
"""

import os
import logging
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from rag_engine import RAGEngine
from chromadb.utils.embedding_functions import MistralEmbeddingFunction


logger = logging.getLogger(__name__)
class EmbeddingWrapper:
        def __init__(self, embedding_fn):
            self.embedding_fn = embedding_fn

        def embed_query(self, query):
            # Assumes query is a single string
            return self.embedding_fn([query])[0]

        def embed_documents(self, documents):
            # Assumes documents is a list of strings
            return self.embedding_fn(documents)

class EmbeddingManager:
    """Manages embedding generation and configuration."""
    
    embedding_model = None

    def __init__(self, embeddings_config: Dict[str, Any]):
        """
        Initialize embedding manager.
        
        Args:
            embeddings_config: Configuration dictionary for embeddings
        """
        self.config = embeddings_config
        self.embedding_function = None
        self.embedding_model = None
        self._setup_embeddings()

    
    def _setup_embeddings(self):
        """Setup embedding function based on configuration."""
        provider = self.config.get('provider', 'mistral')

        if provider == 'mistral':
            self.embedding_model = "mistral-embed"
            raw_embedding_fn = MistralEmbeddingFunction(model=self.embedding_model)
            self.embedding_function = EmbeddingWrapper(raw_embedding_fn)
            
        
            
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def get_embedding_function(self):
        """Get the configured embedding function."""
        if self.embedding_function is None:
            raise ValueError("Embedding function not initialized")
        return self.embedding_function
    
    def embed_query(self, query: str) -> list:
        """
        Embed a single query.
        
        Args:
            query (str): Query text to embed
            
        Returns:
            list: Query embedding vector
        """
        return self.embedding_model.encode(query)
    
    def embed_documents(self, documents: list) -> list:
        """
        Embed multiple documents.
        
        Args:
            documents (list): List of document texts
            
        Returns:
            Tuple: List of embedding vectors
        """
        return self.embedding_model.encode(documents)
    
    