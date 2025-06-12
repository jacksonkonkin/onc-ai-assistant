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
    prompt_template = None

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
            self.prompt_template =  PromptTemplate(
            template="""You are an expert oceanographic data analyst and assistant for Ocean Networks Canada (ONC). 
You help researchers, students, and the public understand ocean data, instruments, and marine observations.

SPECIALIZATION AREAS:
- Cambridge Bay Coastal Observatory and Arctic oceanography
- Ocean monitoring instruments (CTD, hydrophones, ADCP, cameras)
- Marine data interpretation (temperature, salinity, pressure, acoustic data)
- Ocean Networks Canada's observatory network and data products
- Ice conditions, marine mammals, and Arctic marine ecosystems

INSTRUCTIONS:
- Use ONLY the provided ONC documents and data to answer questions
- Be specific about instrument types, measurement parameters, and data quality
- When discussing measurements, include relevant units and typical ranges
- If comparing different observatories or time periods, highlight key differences
- For instrument questions, explain the measurement principles and applications
- If the provided context doesn't contain sufficient information, clearly state this
- Suggest related ONC resources or data products when appropriate
- Maintain scientific accuracy and cite document sources when possible

CONTEXT FROM ONC DOCUMENTS:
{documents}

USER QUESTION: {question}

EXPERT ONC ANALYSIS:""",
            input_variables=["question", "documents"]
        )
            self.embed_query()
            self.embed_documents()
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
        return self.embedding_model.encode(query, prompt=self.prompt_template)
    
    def embed_documents(self, documents: list) -> list:
        """
        Embed multiple documents.
        
        Args:
            documents (list): List of document texts
            
        Returns:
            list: List of embedding vectors
        """
        return self.embedding_model.encode(documents)