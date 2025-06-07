"""
Embedding management and generation
Teams: Data team + LLM team
"""

import os
import logging
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from rag_engine import RAGEngine

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation and configuration."""
    
    embedding_model = None
    prompt_template = RAGEngine._setup_prompt_templates()

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
        provider = self.config.get('provider', 'mistral')
        
        if provider == 'mistral':
            self.embedding_model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")
            print(self.prompt_template)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
    
    def embed_query(self, query: str) -> list:
        """
        Embed a single query.
        
        Args:
            query (str): Query text to embed
            
        Returns:
            list: Query embedding vector
        """
        return self.embedded_model.encode(query, )
    
    def embed_documents(self, documents: list) -> list:
        """
        Embed multiple documents.
        
        Args:
            documents (list): List of document texts
            
        Returns:
            list: List of embedding vectors
        """
        return self.embedding_function.embed_documents(documents)