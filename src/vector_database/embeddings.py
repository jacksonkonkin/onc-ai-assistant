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


logger = logging.getLogger(__name__)


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

   # def Mistral_embeddings(self):
    #     return self.embed_query("what is today?"), self.embed_documents(["apple", "banana", "Tuesday is today"])
    
    def _setup_embeddings(self):
        """Setup embedding function based on configuration."""
        provider = self.config.get('provider', 'mistral')
        
        if provider == 'mistral':
            self.embedding_model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")
            self.embedding_function = self.embed_query
        
            
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