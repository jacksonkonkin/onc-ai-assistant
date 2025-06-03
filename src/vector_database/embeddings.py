"""
Embedding management and generation
Teams: Data team + LLM team
"""

import os
import logging
from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation and configuration."""
    
    def __init__(self, embeddings_config: Dict[str, Any]):
        """
        Initialize embedding manager.
        
        Args:
            embeddings_config: Configuration dictionary for embeddings
        """
        self.config = embeddings_config
        self.embedding_function = None
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Setup embedding function based on configuration."""
        provider = self.config.get('provider', 'openai')
        
        if provider == 'openai':
            self._setup_openai_embeddings()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def _setup_openai_embeddings(self):
        """Setup OpenAI embeddings."""
        api_key_env = self.config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        
        model = self.config.get('model', 'text-embedding-ada-002')
        
        self.embedding_function = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model
        )
        
        logger.info(f"Initialized OpenAI embeddings with model: {model}")
    
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
        return self.embedding_function.embed_query(query)
    
    def embed_documents(self, documents: list) -> list:
        """
        Embed multiple documents.
        
        Args:
            documents (list): List of document texts
            
        Returns:
            list: List of embedding vectors
        """
        return self.embedding_function.embed_documents(documents)