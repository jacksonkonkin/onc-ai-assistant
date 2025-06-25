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
from ..database_search import OceanQuerySystem
from ..rag_engine import RAGEngine, LLMWrapper
from ..conversation import ConversationManager
from ..query_refinement import QueryRefinementManager
from ..feedback import feedback_logger

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
        self.ocean_query_system = None
        self.llm_wrapper = None
        self.rag_engine = None
        self.conversation_manager = None
        self.query_refinement_manager = None
        self.feedback_logger = None
        
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
        print(embeddings_config)
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
        routing_config = self.config_manager.get('query_routing', {})
        self.query_router = QueryRouter(routing_config)
        
        # Ocean Query System with enhanced formatting
        ocean_config = self.config_manager.get('ocean_responses', {})
        enhanced_formatting_enabled = ocean_config.get('enhanced_formatting', False)
        
        if enhanced_formatting_enabled:
            self.ocean_query_system = OceanQuerySystem(llm_wrapper=self.llm_wrapper)
            logger.info("Ocean Query System initialized with enhanced formatting")
        else:
            self.ocean_query_system = OceanQuerySystem()
            logger.info("Ocean Query System initialized with standard formatting")
        
        # Conversation Manager
        conversation_config = self.config_manager.get('conversation', {})
        max_history = conversation_config.get('max_history_length', 10)
        context_window = conversation_config.get('context_window_minutes', 30)
        self.conversation_manager = ConversationManager(
            max_history_length=max_history,
            context_window_minutes=context_window
        )
        
        # Query Refinement Manager
        refinement_config = self.config_manager.get('query_refinement', {})
        self.query_refinement_manager = QueryRefinementManager(
            llm_wrapper=self.llm_wrapper,
            config=refinement_config
        )
        
        # Feedback Logger
        feedback_config = self.config_manager.get('feedback', {})
        self.feedback_logger = feedback_logger.create_feedback_logger(feedback_config)
        
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
            
            # Get conversation context and detect follow-ups FIRST
            conversation_context = ""
            follow_up_info = {}
            
            if self.conversation_manager:
                # Add user message to conversation
                self.conversation_manager.add_user_message(question, context)
                
                # Get conversation context
                conversation_context = self.conversation_manager.get_conversation_context(include_metadata=True)
                
                # Detect follow-up questions
                follow_up_info = self.conversation_manager.detect_follow_up_question(question)
                
                logger.debug(f"Follow-up detection: {follow_up_info}")
            
            # Step 1: Analyze query clarity (check for ambiguity) WITH conversation context
            query_analysis = None
            if self.query_refinement_manager:
                # Include conversation context in the analysis
                analysis_context = context or {}
                analysis_context['conversation_context'] = conversation_context
                
                query_analysis = self.query_refinement_manager.analyze_query_clarity(question, analysis_context)
                
                # If query is highly ambiguous, request clarification
                if self.query_refinement_manager.should_request_clarification(query_analysis):
                    clarification_request = self.query_refinement_manager.format_clarification_request(query_analysis)
                    # Add the clarification to conversation history before returning
                    if self.conversation_manager:
                        self.conversation_manager.add_assistant_message(clarification_request, {
                            'type': 'clarification_request',
                            'clarity_level': query_analysis.clarity_level.value,
                            'missing_parameters': query_analysis.missing_data_parameters
                        })
                    return clarification_request
            
            # Route the query with conversation context
            routing_context = context or {}
            routing_context.update({
                'has_vector_store': self.vector_store_ready,
                'has_database': self.ocean_query_system is not None,
                'conversation_context': conversation_context,
                'follow_up_info': follow_up_info
            })
            
            routing_decision = self.query_router.route_query(question, routing_context)
            query_type = routing_decision['type']
            
            # Step 2: Process query and get initial results
            initial_results = []
            if query_type.value == 'vector_search':
                response, initial_results = self._process_vector_query_with_results(question, conversation_context)
            elif query_type.value == 'database_search':
                response, initial_results = self._process_database_query_with_results(question, routing_decision.get('parameters', {}), conversation_context)
            elif query_type.value == 'hybrid_search':
                response, initial_results = self._process_hybrid_query_with_results(question, routing_decision.get('parameters', {}), conversation_context)
            else:  # direct_llm
                response = self._process_direct_query(question, conversation_context)
                initial_results = []  # Direct LLM doesn't have discrete results to analyze
            
            # Step 3: Analyze results and provide suggestions if needed
            final_response = response
            if self.query_refinement_manager and initial_results is not None:
                result_analysis = self.query_refinement_manager.analyze_results(
                    initial_results, question, query_type.value
                )
                
                # Add suggestions if results are too many/few
                if self.query_refinement_manager.should_provide_suggestions(result_analysis):
                    suggestions = self.query_refinement_manager.format_result_suggestions(result_analysis)
                    if suggestions:
                        final_response += f"\n\n{suggestions}"
            
            # Step 4: Add feedback prompt
            if self.query_refinement_manager:
                feedback_prompt = self.query_refinement_manager.generate_feedback_prompt(
                    question, final_response, routing_decision
                )
                feedback_message = self.query_refinement_manager.format_feedback_prompt(feedback_prompt)
                if feedback_message:
                    final_response += feedback_message
            
            # Add assistant response to conversation
            if self.conversation_manager:
                response_metadata = {
                    'route_type': query_type.value,
                    'classification': routing_decision.get('classification', ''),
                    'confidence': routing_decision.get('confidence', ''),
                    'is_follow_up': follow_up_info.get('is_follow_up', False),
                    'had_refinement': query_analysis is not None,
                    'clarity_level': query_analysis.clarity_level.value if query_analysis else 'unknown'
                }
                self.conversation_manager.add_assistant_message(final_response, response_metadata)
            
            return final_response
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error processing your question: {str(e)}"
    
    def _process_vector_query(self, question: str, conversation_context: str = "") -> str:
        """Process query using vector search."""
        if not self.vector_store_ready:
            return self._process_direct_query(question, conversation_context)
        
        # Retrieve documents
        documents = self.vector_store_manager.retrieve_documents(question)
        
        # Generate response with conversation context
        return self.rag_engine.process_rag_query(question, documents, conversation_context)
    
    def _process_database_query(self, question: str, parameters: Dict[str, Any], conversation_context: str = "") -> str:
        """Process query using database search."""
        if not self.ocean_query_system:
            return self._process_direct_query(question, conversation_context)
        
        try:
            # Determine query type based on parameters
            search_type = parameters.get('search_type', 'structured')
            if search_type == "device_discovery":
                query_type = "device_discovery"
            elif search_type == "data_products":
                query_type = "data_products"
            else:
                query_type = "data"
            
            # Use the ocean query system to process the question
            result = self.ocean_query_system.process_query(question, query_type=query_type)
            
            if result["status"] in ["success", "no_devices", "no_products"]:
                # Use enhanced formatting with conversation context
                formatted_response = self.ocean_query_system.format_enhanced_response(result, conversation_context)
                return formatted_response
            elif result["status"] == "no_data":
                # Use enhanced formatting for no data responses too
                return self.ocean_query_system.format_enhanced_response(result, conversation_context)
            else:
                # On error, fall back to direct query
                logger.warning(f"Ocean query failed: {result.get('message', 'Unknown error')}")
                return self._process_direct_query(question, conversation_context)
                
        except Exception as e:
            logger.error(f"Error in database query: {e}")
            return self._process_direct_query(question, conversation_context)
    
    def _process_hybrid_query(self, question: str, parameters: Dict[str, Any], conversation_context: str = "") -> str:
        """Process query using both vector and database search."""
        vector_docs = []
        database_results = []
        
        # Get vector results if available
        if self.vector_store_ready:
            vector_docs = self.vector_store_manager.retrieve_documents(question)
        
        # Get database results if available
        ocean_response = None
        if self.ocean_query_system:
            try:
                db_result = self.ocean_query_system.process_query(question)
                if db_result["status"] == "success":
                    database_results = db_result["data"]
                    # Get the enhanced formatted response from ocean query system
                    ocean_response = self.ocean_query_system.format_enhanced_response(db_result, conversation_context)
                elif db_result["status"] == "no_data":
                    ocean_response = self.ocean_query_system.format_enhanced_response(db_result, conversation_context)
            except Exception as e:
                logger.warning(f"Database query failed in hybrid mode: {e}")
        
        # Generate hybrid response
        if ocean_response:
            # If we got ocean data, prioritize that over hybrid processing
            return ocean_response
        elif vector_docs:
            return self.rag_engine.process_rag_query(question, vector_docs, conversation_context)
        else:
            return self._process_direct_query(question, conversation_context)
    
    def _process_direct_query(self, question: str, conversation_context: str = "") -> str:
        """Process query using direct LLM."""
        return self.rag_engine.process_direct_query(question, conversation_context)
    
    def _process_vector_query_with_results(self, question: str, conversation_context: str = "") -> tuple[str, List[Any]]:
        """Process query using vector search and return both response and raw results."""
        if not self.vector_store_ready:
            return self._process_direct_query(question, conversation_context), []
        
        # Retrieve documents
        documents = self.vector_store_manager.retrieve_documents(question)
        
        # Generate response with conversation context
        response = self.rag_engine.process_rag_query(question, documents, conversation_context)
        
        return response, documents
    
    def _process_database_query_with_results(self, question: str, parameters: Dict[str, Any], conversation_context: str = "") -> tuple[str, List[Any]]:
        """Process query using database search and return both response and raw results."""
        if not self.ocean_query_system:
            return self._process_direct_query(question, conversation_context), []
        
        try:
            # Determine query type based on parameters
            search_type = parameters.get('search_type', 'structured')
            if search_type == "device_discovery":
                query_type = "device_discovery"
            elif search_type == "data_products":
                query_type = "data_products"
            else:
                query_type = "data"
            
            # Use the ocean query system to process the question
            result = self.ocean_query_system.process_query(question, query_type=query_type)
            
            if result["status"] in ["success", "no_devices", "no_products"]:
                # Use enhanced formatting with conversation context
                formatted_response = self.ocean_query_system.format_enhanced_response(result, conversation_context)
                return formatted_response, result.get("data", [])
            elif result["status"] == "no_data":
                # Use enhanced formatting for no data responses too
                formatted_response = self.ocean_query_system.format_enhanced_response(result, conversation_context)
                return formatted_response, []
            else:
                # On error, fall back to direct query
                logger.warning(f"Ocean query failed: {result.get('message', 'Unknown error')}")
                return self._process_direct_query(question, conversation_context), []
                
        except Exception as e:
            logger.error(f"Error in database query: {e}")
            return self._process_direct_query(question, conversation_context), []
    
    def _process_hybrid_query_with_results(self, question: str, parameters: Dict[str, Any], conversation_context: str = "") -> tuple[str, List[Any]]:
        """Process query using both vector and database search and return both response and raw results."""
        vector_docs = []
        database_results = []
        
        # Get vector results if available
        if self.vector_store_ready:
            vector_docs = self.vector_store_manager.retrieve_documents(question)
        
        # Get database results if available
        ocean_response = None
        if self.ocean_query_system:
            try:
                # Determine query type based on parameters
                search_type = parameters.get('search_type', 'structured')
                query_type = "device_discovery" if search_type == "device_discovery" else "data"
                
                db_result = self.ocean_query_system.process_query(question, query_type=query_type)
                if db_result["status"] in ["success", "no_devices"]:
                    database_results = db_result["data"]
                    # Get the enhanced formatted response from ocean query system
                    ocean_response = self.ocean_query_system.format_enhanced_response(db_result, conversation_context)
                elif db_result["status"] == "no_data":
                    ocean_response = self.ocean_query_system.format_enhanced_response(db_result, conversation_context)
            except Exception as e:
                logger.warning(f"Database query failed in hybrid mode: {e}")
        
        # Generate hybrid response
        if ocean_response:
            # If we got ocean data, prioritize that over hybrid processing
            return ocean_response, database_results
        elif vector_docs:
            response = self.rag_engine.process_rag_query(question, vector_docs, conversation_context)
            return response, vector_docs
        else:
            response = self._process_direct_query(question, conversation_context)
            return response, []
    
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
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state."""
        if self.conversation_manager:
            return self.conversation_manager.get_conversation_summary()
        return {"message": "Conversation management not available"}
    
    def clear_conversation(self) -> None:
        """Clear current conversation history."""
        if self.conversation_manager:
            self.conversation_manager.clear_conversation()
            logger.info("Conversation history cleared")
        else:
            logger.warning("Conversation manager not available")
    
    def save_conversation(self, filepath: str) -> bool:
        """Save current conversation to file."""
        if self.conversation_manager:
            return self.conversation_manager.save_conversation(filepath)
        return False
    
    def load_conversation(self, filepath: str) -> bool:
        """Load conversation from file."""
        if self.conversation_manager:
            return self.conversation_manager.load_conversation(filepath)
        return False
    
    def rate_response(self, rating: str, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Rate a response with thumbs up/down feedback.
        
        Args:
            rating: Either 'thumbs_up' or 'thumbs_down'
            query: The original user query
            response: The system response that was rated
            metadata: Additional context about the response
            
        Returns:
            bool: True if rating was logged successfully
        """
        if not self.feedback_logger:
            logger.warning("Feedback logger not available")
            return False
        
        try:
            if rating.lower() in ['thumbs_up', '👍', 'up', 'good', 'yes']:
                feedback_rating = feedback_logger.FeedbackRating.THUMBS_UP
            elif rating.lower() in ['thumbs_down', '👎', 'down', 'bad', 'no']:
                feedback_rating = feedback_logger.FeedbackRating.THUMBS_DOWN
            else:
                logger.warning(f"Invalid rating: {rating}")
                return False
            
            success = self.feedback_logger.log_feedback(
                rating=feedback_rating,
                query=query,
                response=response,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Logged {feedback_rating.value} feedback")
            
            return success
            
        except Exception as e:
            logger.error(f"Error rating response: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_logger:
            return {"error": "Feedback logger not available"}
        
        return self.feedback_logger.get_feedback_stats()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        status = {
            "setup_complete": self.is_setup,
            "documents_loaded": self.documents_loaded,
            "document_count": len(self.documents) if hasattr(self, 'documents') else 0,
            "vector_store_ready": self.vector_store_ready,
            "vector_document_count": self.vector_store_manager.get_document_count() if self.vector_store_manager else 0,
            "database_connected": self.ocean_query_system is not None,
            "llm_info": self.llm_wrapper.get_model_info() if self.llm_wrapper else {},
            "config": self.config_manager.config
        }
        
        # Add feedback stats if available
        if self.feedback_logger:
            status["feedback_stats"] = self.get_feedback_stats()
        
        return status
    
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
        print("Commands: 'quit' to exit, 'clear' to clear conversation, 'status' for conversation summary")
        print("Feedback: After responses, you can rate with '👍' or '👎'")
        if self.conversation_manager:
            print("💬 Conversation memory: ENABLED - Follow-up questions supported!")
        else:
            print("⚠️  Conversation memory: DISABLED")
        print("="*70 + "\n")
        
        question_count = 0
        last_query = ""
        last_response = ""
        
        while True:
            try:
                question = input("Ask about ONC data: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print(f"\nThank you for using the ONC Assistant!")
                    if question_count > 0:
                        print(f"You asked {question_count} question{'s' if question_count != 1 else ''}.")
                    
                    # Show conversation summary if available
                    if self.conversation_manager and question_count > 0:
                        summary = self.get_conversation_summary()
                        print(f"Conversation summary: {summary.get('message_count', 0)} messages, "
                              f"{summary.get('session_duration_minutes', 0):.1f} minutes")
                    break
                
                # Handle conversation management commands
                if question.lower() == 'clear':
                    self.clear_conversation()
                    print("🗑️  Conversation history cleared. Starting fresh!")
                    question_count = 0
                    continue
                
                if question.lower() == 'status':
                    if self.conversation_manager:
                        summary = self.get_conversation_summary()
                        print(f"💬 Conversation Status:")
                        print(f"   Messages: {summary.get('message_count', 0)}")
                        print(f"   Duration: {summary.get('session_duration_minutes', 0):.1f} minutes")
                        print(f"   Topics: {', '.join(summary.get('topics_discussed', [])[:5])}")
                        print(f"   Data queries: {summary.get('data_queries_made', 0)}")
                    else:
                        print("⚠️  Conversation management not available")
                    continue
                
                # Handle feedback for previous response
                if question.strip() in ['👍', 'thumbs_up', 'up', '👎', 'thumbs_down', 'down']:
                    if last_query and last_response:
                        success = self.rate_response(question.strip(), last_query, last_response)
                        if success:
                            if question.strip() in ['👍', 'thumbs_up', 'up']:
                                print("👍 Thanks for the positive feedback!")
                            else:
                                print("👎 Thanks for the feedback. We'll work to improve!")
                        else:
                            print("Sorry, couldn't save your feedback.")
                    else:
                        print("No previous response to rate.")
                    continue
                
                if not question:
                    continue
                
                question_count += 1
                
                # Show follow-up detection if available
                if self.conversation_manager and question_count > 1:
                    follow_up_info = self.conversation_manager.detect_follow_up_question(question)
                    if follow_up_info.get('is_follow_up') and follow_up_info.get('confidence', 0) > 0.6:
                        print("🔗 Follow-up question detected - using conversation context...")
                
                print("Processing query...", end='', flush=True)
                
                answer = self.query(question)
                
                print("\r" + " "*20 + "\r", end='')
                print(f"**ONC Assistant:** {answer}\n")
                print("─" * 50)
                
                # Store for potential feedback
                last_query = question
                last_response = answer
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {str(e)}")
                print("Please try again.\n")