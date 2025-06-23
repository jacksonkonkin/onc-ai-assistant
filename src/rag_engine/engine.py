"""
Core RAG engine implementation
Team: LLM team
"""

import logging
from typing import Dict, Any, List, Optional

from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)


class RAGEngine:
    """Core RAG engine for processing queries and generating responses."""
    
    def __init__(self, llm_wrapper: LLMWrapper):
        """
        Initialize RAG engine.
        
        Args:
            llm_wrapper: Configured LLM wrapper
        """
        self.llm_wrapper = llm_wrapper
        self.rag_chain = None
        self.direct_chain = None
        
        self._setup_prompt_templates()
    
    def _setup_prompt_templates(self):
        """Setup prompt templates for different modes."""
        # RAG mode template (with documents)
        self.rag_prompt = PromptTemplate(
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
- If this is a follow-up question, reference previous conversation context appropriately

{conversation_history}

CONTEXT FROM ONC DOCUMENTS:
{documents}

USER QUESTION: {question}

EXPERT ONC ANALYSIS:""",
            input_variables=["question", "documents", "conversation_history"]
        )
        
        # Direct mode template (without documents)
        self.direct_prompt = PromptTemplate(
            template="""You are an expert oceanographic data analyst and assistant for Ocean Networks Canada (ONC). 
You help researchers, students, and the public understand ocean data, instruments, and marine observations.

SPECIALIZATION AREAS:
- Cambridge Bay Coastal Observatory and Arctic oceanography
- Ocean monitoring instruments (CTD, hydrophones, ADCP, cameras)
- Marine data interpretation (temperature, salinity, pressure, acoustic data)
- Ocean Networks Canada's observatory network and data products
- Ice conditions, marine mammals, and Arctic marine ecosystems

INSTRUCTIONS:
- Draw from your knowledge of oceanography and marine science to answer questions
- Be specific about instrument types, measurement parameters, and data quality when applicable
- When discussing measurements, include relevant units and typical ranges
- Focus on ONC-specific context when possible (Cambridge Bay, Arctic oceanography, etc.)
- For instrument questions, explain the measurement principles and applications
- If you don't have specific information, clearly state this and suggest consulting ONC resources
- Suggest related ONC data products or documentation when appropriate
- Maintain scientific accuracy and provide educational value
- If this is a follow-up question, reference previous conversation context appropriately

{conversation_history}

NOTE: No specific ONC documents are currently loaded, so responses are based on general oceanographic knowledge.

USER QUESTION: {question}

EXPERT ONC ANALYSIS:""",
            input_variables=["question", "conversation_history"]
        )

    
    def setup_rag_mode(self):
        """Setup RAG processing chain."""
        def rag_chain(inputs):
            formatted_prompt = self.rag_prompt.format(**inputs)
            response = self.llm_wrapper.invoke(formatted_prompt)
            return response
        
        self.rag_chain = rag_chain
        logger.info("RAG mode chain initialized")
    
    def setup_direct_mode(self):
        """Setup direct LLM processing chain."""
        def direct_chain(inputs):
            formatted_prompt = self.direct_prompt.format(**inputs)
            response = self.llm_wrapper.invoke(formatted_prompt)
            return response
        
        self.direct_chain = direct_chain
        logger.info("Direct mode chain initialized")
    
    def process_rag_query(self, question: str, documents: List[Document], 
                         conversation_context: str = "") -> str:
        """
        Process query using RAG mode with documents.
        
        Args:
            question: User question
            documents: Retrieved documents
            conversation_context: Previous conversation context
            
        Returns:
            Generated response
        """
        if not self.rag_chain:
            self.setup_rag_mode()
        
        try:
            # Format documents for the prompt
            formatted_docs = self._format_documents(documents)
            
            # Format conversation context
            formatted_conversation = self._format_conversation_context(conversation_context)
            
            # Generate response
            response = self.rag_chain({
                "question": question,
                "documents": formatted_docs,
                "conversation_history": formatted_conversation
            })
            
            logger.info("RAG query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return f"Sorry, I encountered an error processing your question: {str(e)}"
    
    def process_direct_query(self, question: str, conversation_context: str = "") -> str:
        """
        Process query using direct LLM mode.
        
        Args:
            question: User question
            conversation_context: Previous conversation context
            
        Returns:
            Generated response
        """
        if not self.direct_chain:
            self.setup_direct_mode()
        
        try:
            # Format conversation context
            formatted_conversation = self._format_conversation_context(conversation_context)
            
            response = self.direct_chain({
                "question": question,
                "conversation_history": formatted_conversation
            })
            logger.info("Direct query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing direct query: {e}")
            return f"Sorry, I encountered an error processing your question: {str(e)}"
    
    def process_hybrid_query(self, question: str, vector_docs: List[Document], 
                           database_results: List[Dict[str, Any]], 
                           conversation_context: str = "") -> str:
        """
        Process query using hybrid mode (vector + database results).
        
        Args:
            question: User question
            vector_docs: Documents from vector search
            database_results: Results from database search
            conversation_context: Previous conversation context
            
        Returns:
            Generated response
        """
        if not self.rag_chain:
            self.setup_rag_mode()
        
        try:
            # Combine vector documents and database results
            combined_context = self._combine_contexts(vector_docs, database_results)
            
            # Format conversation context
            formatted_conversation = self._format_conversation_context(conversation_context)
            
            # Generate response
            response = self.rag_chain({
                "question": question,
                "documents": combined_context,
                "conversation_history": formatted_conversation
            })
            
            logger.info("Hybrid query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing hybrid query: {e}")
            return f"Sorry, I encountered an error processing your question: {str(e)}"
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for inclusion in prompts."""
        if not documents:
            return "No relevant documents found."
        
        doc_texts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get('filename', f'Document_{i+1}')
            doc_type = doc.metadata.get('doc_type', 'unknown')
            
            header = f"[{source}] (Format: {doc_type})"
            doc_texts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n" + "="*60 + "\n\n".join(doc_texts)
    
    def _combine_contexts(self, vector_docs: List[Document], 
                         database_results: List[Dict[str, Any]]) -> str:
        """Combine vector documents and database results into unified context."""
        combined_context = []
        
        # Add vector documents
        if vector_docs:
            doc_context = self._format_documents(vector_docs)
            combined_context.append(f"DOCUMENT SOURCES:\n{doc_context}")
        
        # Add database results
        if database_results:
            db_context = self._format_database_results(database_results)
            combined_context.append(f"DATABASE RESULTS:\n{db_context}")
        
        return "\n\n" + "="*80 + "\n\n".join(combined_context)
    
    def _format_database_results(self, database_results: List[Dict[str, Any]]) -> str:
        """Format database results for inclusion in prompts."""
        if not database_results:
            return "No database results found."
        
        formatted_results = []
        for i, result in enumerate(database_results):
            result_text = f"Result {i+1}:\n"
            for key, value in result.items():
                result_text += f"  {key}: {value}\n"
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results)
    
    def _format_conversation_context(self, conversation_context: str) -> str:
        """
        Format conversation context for inclusion in prompts.
        
        Args:
            conversation_context: Raw conversation context string
            
        Returns:
            Formatted conversation context
        """
        if not conversation_context or not conversation_context.strip():
            return ""
        
        # If already formatted, return as is
        if conversation_context.startswith("Previous conversation context:"):
            return conversation_context
        
        # Otherwise, format it properly
        return f"""
CONVERSATION CONTEXT:
{conversation_context}

Instructions: Reference the above conversation when answering the current question. If this appears to be a follow-up question, provide context-aware responses that build on previous answers.
"""
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and configuration."""
        return {
            "rag_chain_ready": self.rag_chain is not None,
            "direct_chain_ready": self.direct_chain is not None,
            "llm_info": self.llm_wrapper.get_model_info()
        }