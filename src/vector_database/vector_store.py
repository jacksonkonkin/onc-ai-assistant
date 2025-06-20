"""
Vector store management and operations
Teams: Data team + LLM team
"""

import logging
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb


from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations and document retrieval."""
    
    def __init__(self, vector_config: Dict[str, Any], 
                 processing_config: Dict[str, Any],
                 embedding_manager: EmbeddingManager):
        """
        Initialize vector store manager.
        
        Args:
            vector_config: Vector store configuration
            processing_config: Document processing configuration
            embedding_manager: Configured embedding manager
        """
        self.vector_config = vector_config
        self.processing_config = processing_config
        self.embedding_manager = embedding_manager
        self.vectorstore = None
        self.retriever = None
        self.text_splitter = None
        
        self._setup_text_splitter()
    
    def _setup_text_splitter(self):
        """Setup text splitter for document chunks."""
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.processing_config.get('chunk_size', 500),
            chunk_overlap=self.processing_config.get('chunk_overlap', 50),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def setup_vectorstore(self, documents: List[Document]) -> bool:
        """
        Setup vector store with documents.
        
        Args:
            documents: List of documents to index
            
        Returns:
            bool: True if successful
        """
        try:
            # Split documents into chunks
            doc_splits = self._split_documents(documents)
            
            if not doc_splits:
                logger.warning("No document chunks created")
                return False
            
            # Setup vector store
            self._create_vectorstore(doc_splits)
            
            # Setup retriever
            self._setup_retriever()
            
            logger.info("Vector store setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            return False
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            return []
        
        doc_splits = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(doc_splits)} document chunks")
        return doc_splits
    
    def _create_vectorstore(self, doc_splits: List[Document]):
        """Create or load vector store."""
        persist_dir = Path(self.vector_config.get('persist_directory', 'onc_vectorstore'))
        collection_name = self.vector_config.get('collection_name', 'onc_documents')
        force_rebuild = self.vector_config.get('force_rebuild', False)
        
        embedding_function = self.embedding_manager.get_embedding_function()
        
        # Check if persistent store exists
        if (persist_dir.exists() and 
            not force_rebuild and
            len(list(persist_dir.iterdir())) > 0):
            
            logger.info(f"Loading existing vector store from {persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embedding_function,
                collection_name=collection_name
            )
            logger.info(f"Loaded {self.vectorstore._collection.count()} documents")
        else:
            # Create new vector store
            logger.info("Creating new persistent vector store...")
            persist_dir.mkdir(exist_ok=True)
            
            batch_size = self.processing_config.get('batch_size', 20)
            self.vectorstore = self._create_chroma_vectorstore(
                doc_splits, embedding_function, batch_size, 
                str(persist_dir), collection_name
            )
    
    def _create_chroma_vectorstore(self, doc_splits: List[Document], 
                                 embedding_function, batch_size: int,
                                 persist_directory: str, collection_name: str):
        """Create Chroma vector store with batched processing."""
        logger.info(f"Creating Chroma vector store with {len(doc_splits)} documents")
        
        # Create initial vectorstore with first batch
        first_batch = doc_splits[:batch_size]
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Process remaining batches
        for i in range(batch_size, len(doc_splits), batch_size):
            batch = doc_splits[i:i + batch_size]
            
            try:
                vectorstore.add_documents(batch)
                logger.info(f"✓ Added batch {i//batch_size + 1}/{(len(doc_splits)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"✗ Error in batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Chroma vector store created with {vectorstore._collection.count()} documents")
        return vectorstore
    
    def _setup_retriever(self):
        """Setup document retriever."""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return
        
        retrieval_config = self.vector_config.get('retrieval', {})
        k = retrieval_config.get('k', 8)
        
        # Adjust k based on available documents
        doc_count = self.vectorstore._collection.count() if hasattr(self.vectorstore, '_collection') else 0
        max_k = min(k, doc_count) if doc_count > 0 else k
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": max_k}
        )
        
        logger.info(f"Retriever setup with k={max_k}")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Document]: Retrieved documents
        """
        if not self.retriever:
            logger.error("Retriever not initialized")
            return []
        
        try:
            documents = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def add_documents(self, new_documents: List[Document]) -> bool:
        """
        Add new documents to existing vector store.
        
        Args:
            new_documents: Documents to add
            
        Returns:
            bool: True if successful
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return False
        
        if not new_documents:
            logger.info("No new documents to add")
            return True
        
        try:
            # Split new documents
            doc_splits = self._split_documents(new_documents)
            
            # Add to vector store
            self.vectorstore.add_documents(doc_splits)
            logger.info(f"✓ Successfully added {len(doc_splits)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error adding documents: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in vector store."""
        if not self.vectorstore or not hasattr(self.vectorstore, '_collection'):
            return 0
        return self.vectorstore._collection.count()