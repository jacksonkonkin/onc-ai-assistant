"""
Main pipeline orchestrator - integrates all modules
Teams: All teams (integration point)
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..config.settings import ConfigManager
from ..document_processing import DocumentProcessor, DocumentLoader
from ..vector_database import VectorStoreManager, EmbeddingManager
from ..query_routing import QueryRouter
from ..database_search import Ocean3Client
from ..rag_engine import RAGEngine, LLMWrapper

logger = logging.getLogger(__name__)


class ONCPipeline:
    """
    Main ONC RAG Pipeline that orchestrates all modules.
    
    This class serves as the integration point for all teams and modules.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ONC Pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        
        # Initialize components
        self.document_loader = None
        self.document_processor = None
        self.embedding_manager = None
        self.vector_store_manager = None
        self.query_router = None
        self.ocean3_client = None
        self.llm_wrapper = None
        self.rag_engine = None
        
        # State tracking
        self.is_setup = False
        self.documents_loaded = False
        self.vector_store_ready = False
        
        logger.info("ONC Pipeline initialized")
    
    def setup(self, doc_dir: Optional[str] = None, force_rebuild: bool = False) -> bool:
        """
        Setup the complete pipeline.
        
        Args:
            doc_dir: Document directory path
            force_rebuild: Force rebuild of vector store
            
        Returns:
            bool: True if setup successful
        """
        try:
            logger.info("Starting ONC Pipeline setup...")
            
            # Step 1: Initialize core components
            self._setup_core_components()
            
            # Step 2: Load and process documents
            success = self._setup_documents(doc_dir)
            
            # Step 3: Setup vector store if documents available
            if success and self.documents_loaded:
                self._setup_vector_store(force_rebuild)
            
            # Step 4: Setup remaining components
            self._setup_additional_components()
            
            self.is_setup = True
            logger.info("ONC Pipeline setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during pipeline setup: {e}")
            return False
    
    def _setup_core_components(self):
        """Setup core processing components."""
        # Document processing
        self.document_loader = DocumentLoader(
            self.config_manager.get('document_dir', 'onc_documents')
        )
        self.document_processor = DocumentProcessor()
        
        # Embeddings
        embeddings_config = self.config_manager.get_embeddings_config()
        self.embedding_manager = EmbeddingManager(embeddings_config)
        
        # LLM
        llm_config = self.config_manager.get_llm_config()
        self.llm_wrapper = LLMWrapper(llm_config)
        
        # RAG Engine
        self.rag_engine = RAGEngine(self.llm_wrapper)
        
        logger.info("Core components initialized")
    
    def _setup_documents(self, doc_dir: Optional[str]) -> bool:
        """Setup document loading and processing."""
        try:
            # Load documents
            if doc_dir:
                document_files = self.document_loader.load_local_documents(doc_dir)
            else:
                document_files = self.document_loader.load_local_documents()
            
            if not document_files:
                logger.warning("No documents found - will use direct LLM mode")
                self.documents_loaded = False
                return True
            
            # Process documents
            logger.info("Processing documents...")
            self.documents = self.document_processor.process_documents(document_files)
            
            if self.documents:
                self.documents_loaded = True
                logger.info(f"Successfully processed {len(self.documents)} documents")
                return True
            else:
                logger.warning("No documents successfully processed")
                self.documents_loaded = False
                return True
                
        except Exception as e:
            logger.error(f"Error setting up documents: {e}")
            return False
    
    def _setup_vector_store(self, force_rebuild: bool):
        """Setup vector store and retrieval."""
        try:
            processing_config = self.config_manager.get_processing_config()
            vector_config = self.config_manager.get_vector_store_config()
            
            if force_rebuild:
                vector_config['force_rebuild'] = True
            
            self.vector_store_manager = VectorStoreManager(
                vector_config, processing_config, self.embedding_manager
            )
            
            # Setup with documents
            success = self.vector_store_manager.setup_vectorstore(self.documents)
            
            if success:
                self.vector_store_ready = True
                logger.info("Vector store setup completed")
            else:
                logger.warning("Vector store setup failed")
                
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
    
    def _setup_additional_components(self):
        """Setup query routing and database components."""
        # Query router
        routing_config = self.config_manager.get('routing', {})
        self.query_router = QueryRouter(routing_config)
        
        # Ocean 3 database client
        db_config = self.config_manager.get('database', {})
        self.ocean3_client = Ocean3Client(db_config)
        
        # Setup RAG engine modes
        if self.vector_store_ready:
            self.rag_engine.setup_rag_mode()
        else:
            self.rag_engine.setup_direct_mode()
        
        logger.info("Additional components initialized")
    
    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a user query through the complete pipeline.
        
        Args:
            question: User question
            context: Additional context for routing
            
        Returns:
            Generated response
        """
        if not self.is_setup:
            raise ValueError("Pipeline not setup. Call setup() first.")
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Route the query
            routing_context = context or {}
            routing_context.update({
                'has_vector_store': self.vector_store_ready,
                'has_database': self.ocean3_client.is_connected()
            })
            
            routing_decision = self.query_router.route_query(question, routing_context)
            query_type = routing_decision['type']
            
            # Process based on routing decision
            if query_type.value == 'vector_search':
                return self._process_vector_query(question)
            elif query_type.value == 'database_search':
                return self._process_database_query(question, routing_decision.get('parameters', {}))
            elif query_type.value == 'hybrid_search':
                return self._process_hybrid_query(question, routing_decision.get('parameters', {}))
            else:  # direct_llm
                return self._process_direct_query(question)
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error processing your question: {str(e)}"
    
    def _process_vector_query(self, question: str) -> str:
        """Process query using vector search."""
        if not self.vector_store_ready:
            return self._process_direct_query(question)
        
        # Retrieve documents
        documents = self.vector_store_manager.retrieve_documents(question)
        
        # Generate response
        return self.rag_engine.process_rag_query(question, documents)
    
    def _process_database_query(self, question: str, parameters: Dict[str, Any]) -> str:
        """Process query using database search."""
        if not self.ocean3_client.is_connected():
            return self._process_direct_query(question)
        
        # TODO: Implement database query logic
        # This will be expanded by Backend/Data teams
        
        # For now, fall back to direct query
        return self._process_direct_query(question)
    
    def _process_hybrid_query(self, question: str, parameters: Dict[str, Any]) -> str:
        """Process query using both vector and database search."""
        vector_docs = []
        database_results = []
        
        # Get vector results if available
        if self.vector_store_ready:
            vector_docs = self.vector_store_manager.retrieve_documents(question)
        
        # Get database results if available
        if self.ocean3_client.is_connected():
            # TODO: Implement database query
            pass
        
        # Generate hybrid response
        if vector_docs or database_results:
            return self.rag_engine.process_hybrid_query(question, vector_docs, database_results)
        else:
            return self._process_direct_query(question)
    
    def _process_direct_query(self, question: str) -> str:
        """Process query using direct LLM."""
        return self.rag_engine.process_direct_query(question)
    
    def add_documents(self, file_paths: List[str]) -> bool:
        """
        Add new documents to the pipeline.
        
        Args:
            file_paths: List of document file paths to add
            
        Returns:
            bool: True if successful
        """
        if not self.is_setup:
            logger.error("Pipeline not setup")
            return False
        
        try:
            # Process new documents
            new_documents = self.document_processor.process_documents(file_paths)
            
            if not new_documents:
                logger.warning("No new documents processed")
                return False
            
            # Add to vector store if available
            if self.vector_store_ready:
                success = self.vector_store_manager.add_documents(new_documents)
                if success:
                    self.documents.extend(new_documents)
                    logger.info(f"Added {len(new_documents)} new documents")
                return success
            else:
                logger.warning("Vector store not available - documents not added")
                return False
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            "setup_complete": self.is_setup,
            "documents_loaded": self.documents_loaded,
            "document_count": len(self.documents) if hasattr(self, 'documents') else 0,
            "vector_store_ready": self.vector_store_ready,
            "vector_document_count": self.vector_store_manager.get_document_count() if self.vector_store_manager else 0,
            "database_connected": self.ocean3_client.is_connected() if self.ocean3_client else False,
            "llm_info": self.llm_wrapper.get_model_info() if self.llm_wrapper else {},
            "config": self.config_manager.config
        }
    
    def interactive_mode(self):
        """Run interactive Q&A mode."""
        if not self.is_setup:
            print("Pipeline not setup. Please call setup() first.")
            return
        
        print("\n" + "="*70)
        print("ONC OCEAN DATA ASSISTANT")
        print("Modular AI Assistant for Ocean Networks Canada")
        print("="*70)
        
        # Show status
        status = self.get_pipeline_status()
        print(f"Documents loaded: {status['document_count']}")
        print(f"Vector store: {'Ready' if status['vector_store_ready'] else 'Not available'}")
        print(f"Database: {'Connected' if status['database_connected'] else 'Not connected'}")
        print(f"LLM: {status['llm_info'].get('provider', 'Unknown')} - {status['llm_info'].get('model', 'Unknown')}")
        
        print("-"*70)
        print("Ask about Ocean Networks Canada data and instruments")
        print("Type 'quit' to exit")
        print("="*70 + "\n")
        
        question_count = 0
        
        while True:
            try:
                question = input("Ask about ONC data: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print(f"\nThank you for using the ONC Assistant!")
                    if question_count > 0:
                        print(f"You asked {question_count} question{'s' if question_count != 1 else ''}.")
                    break
                
                question_count += 1
                print("Processing query...", end='', flush=True)
                
                answer = self.query(question)
                
                print("\r" + " "*20 + "\r", end='')
                print(f"**ONC Assistant:** {answer}\n")
                print("─" * 50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {str(e)}")
                print("Please try again.\n")