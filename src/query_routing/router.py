"""
Query routing logic and decision making
Teams: Backend team + Data team
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import asyncio
import os
from groq import Groq
from transformers import BertForSequenceClassification, BertTokenizer
from huggingface_hub import hf_hub_download
import torch
import pickle

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    VECTOR_SEARCH = "vector_search"
    DATABASE_SEARCH = "database_search"
    HYBRID_SEARCH = "hybrid_search"
    DIRECT_LLM = "direct_llm"


class QueryRouter:
    """Routes queries to appropriate data sources and processing pipelines."""
    
    def __init__(self, routing_config: Optional[Dict[str, Any]] = None):
        """
        Initialize query router.
        
        Args:
            routing_config: Configuration for routing logic
        """
        self.config = routing_config or {}
        self.vector_keywords = self._load_vector_keywords()
        self.database_keywords = self._load_database_keywords()
        
        # Initialize BERT model for query classification
        self.bert_model = None
        self.bert_tokenizer = None
        self.label_encoder = None
        
        try:
            repo_id = "kgosal03/bert-query-classifier"
            self.bert_model = BertForSequenceClassification.from_pretrained(repo_id)
            self.bert_tokenizer = BertTokenizer.from_pretrained(repo_id)
            
            # Load label encoder
            label_path = hf_hub_download(repo_id, filename="label_encoder.pkl")
            with open(label_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            
            # Set model to evaluation mode
            self.bert_model.eval()
            logger.info("BERT-based query routing enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize BERT model: {e}")
        
        # Initialize LLM client for fallback routing
        self.groq_client = None
        if os.getenv('GROQ_API_KEY'):
            try:
                self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                logger.info("LLM fallback routing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
        
        # Enable/disable BERT routing (prefer BERT over LLM)
        self.use_bert_routing = self.config.get('use_bert_routing', True) and self.bert_model is not None
        self.use_llm_routing = self.config.get('use_llm_routing', True) and self.groq_client is not None
    
    def _load_vector_keywords(self) -> List[str]:
        """Load keywords that indicate vector search should be used."""
        default_keywords = [
            "document", "paper", "research", "study", "publication",
            "article", "content", "text", "explain", "describe",
            "what is", "how does", "definition", "overview"
        ]
        return self.config.get('vector_keywords', default_keywords)
    
    def _load_database_keywords(self) -> List[str]:
        """Load keywords that indicate database search should be used."""
        default_keywords = [
            # Ocean parameters
            "temperature", "salinity", "pressure", "depth", "ph", "oxygen", "conductivity",
            "chlorophyll", "turbidity", "density", "fluorescence",
            # Data request terms
            "data", "measurement", "sensor", "instrument", "value", "reading",
            "latest", "current", "recent", "today", "yesterday", "now",
            # Location terms
            "cambridge bay", "station", "location", "coordinates", "site",
            # Time terms  
            "time series", "when", "since", "from", "to", "between",
            # Request patterns
            "get", "show", "find", "retrieve", "what is the", "how much",
            "give me", "tell me"
        ]
        return self.config.get('database_keywords', default_keywords)
    
    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate processing pipeline.
        
        Args:
            query (str): User query
            context (Dict): Additional context for routing decisions
            
        Returns:
            Dict: Routing decision with type and parameters
        """
        context = context or {}
        
        # Check for conversation context and follow-up detection
        conversation_context = context.get('conversation_context', '')
        follow_up_info = context.get('follow_up_info', {})
        
        # Use BERT-based routing if available, otherwise fall back to LLM, then keyword-based
        if self.use_bert_routing:
            try:
                routing_decision = self._bert_route_query(query, context)
                
                # Enhance routing decision with conversation context
                if conversation_context or follow_up_info.get('is_follow_up'):
                    routing_decision = self._enhance_routing_with_context(
                        routing_decision, query, context
                    )
                
                logger.info(f"BERT routed query to: {routing_decision['type']}")
                return routing_decision
            except Exception as e:
                logger.warning(f"BERT routing failed, falling back to LLM: {e}")
        
        # Fallback to LLM-based routing if BERT fails
        if self.use_llm_routing:
            try:
                routing_decision = self._llm_route_query(query, context)
                
                # Enhance routing decision with conversation context
                if conversation_context or follow_up_info.get('is_follow_up'):
                    routing_decision = self._enhance_routing_with_context(
                        routing_decision, query, context
                    )
                
                logger.info(f"LLM routed query to: {routing_decision['type']}")
                return routing_decision
            except Exception as e:
                logger.warning(f"LLM routing failed, falling back to keyword-based: {e}")
        
        # Fallback to keyword-based routing
        query_lower = query.lower()
        vector_score = self._calculate_vector_score(query_lower)
        database_score = self._calculate_database_score(query_lower)
        
        # Adjust scores based on conversation context
        if follow_up_info.get('is_follow_up'):
            vector_score, database_score = self._adjust_scores_for_follow_up(
                vector_score, database_score, follow_up_info
            )
        
        routing_decision = self._make_routing_decision(
            vector_score, database_score, context
        )
        
        logger.info(f"Keyword routed query to: {routing_decision['type']}")
        return routing_decision
    
    def _bert_route_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use BERT model to classify and route the query.
        
        Args:
            query: User query
            context: Additional context for routing
            
        Returns:
            Dict: Routing decision with type and parameters
        """
        # Tokenize input
        inputs = self.bert_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        
        # Predict
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
            
            # Get confidence score (softmax probability)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence = probabilities[0][predicted_class_id].item()
        
        # Decode the label
        predicted_label = self.label_encoder.inverse_transform([predicted_class_id])[0]
        
        logger.debug(f"BERT classified query as: {predicted_label} (confidence: {confidence:.3f})")
        
        # Apply post-processing correction for known BERT model issues
        corrected_label = self._correct_bert_classification(predicted_label, query)
        if corrected_label != predicted_label:
            logger.info(f"Corrected BERT classification from '{predicted_label}' to '{corrected_label}'")
            predicted_label = corrected_label
        
        # Map BERT classification to routing decision
        return self._map_bert_classification_to_route(predicted_label, confidence, context, query)
    
    def _correct_bert_classification(self, predicted_label: str, query: str) -> str:
        """
        Apply post-processing corrections for known BERT model classification issues.
        
        Args:
            predicted_label: Original BERT classification
            query: User query
            
        Returns:
            Corrected classification
        """
        query_lower = query.lower()
        
        # Keywords that strongly indicate data/observation requests
        data_request_keywords = [
            "do you have", "show me", "get me", "give me", "find", "retrieve",
            "data", "measurements", "readings", "values", "observations",
            "temperature data", "salinity data", "pressure data", "ph data",
            "sensor data", "instrument data", "latest", "recent", "current",
            "from yesterday", "from today", "last week", "last month"
        ]
        
        # Temporal patterns that indicate data requests
        temporal_patterns = [
            "last year", "this day last year", "yesterday", "today", "last week",
            "last month", "on this day", "current", "now", "recent", "latest"
        ]
        
        # Parameter patterns that indicate observation queries
        parameter_patterns = [
            "temperature", "salinity", "pressure", "ph", "oxygen", "conductivity",
            "chlorophyll", "turbidity", "density", "fluorescence"
        ]
        
        # If classified as deployment_info but contains strong data request indicators
        if predicted_label == "deployment_info":
            data_score = sum(1 for keyword in data_request_keywords if keyword in query_lower)
            temporal_score = sum(1 for pattern in temporal_patterns if pattern in query_lower)
            parameter_score = sum(1 for param in parameter_patterns if param in query_lower)
            
            # High confidence observation query patterns
            if any(pattern in query_lower for pattern in [
                "what was the", "what is the", "current", "temperature in", "temperature at",
                "temperature on", "data from", "measurements from", "readings from"
            ]):
                return "observation_query"
            
            # If query has temporal references + parameters = observation query
            if temporal_score >= 1 and parameter_score >= 1:
                return "observation_query"
            
            # If query has multiple data request indicators, likely an observation query
            if data_score >= 2:
                return "observation_query"
        
        # If classified as general_knowledge but contains specific data/time requests
        elif predicted_label == "general_knowledge":
            if any(pattern in query_lower for pattern in [
                "measurements from", "data from", "yesterday", "last week",
                "show me", "get me", "give me data"
            ]):
                return "observation_query"
        
        return predicted_label
    
    def _map_bert_classification_to_route(self, classification: str, confidence: float, 
                                        context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Map BERT classification to routing decision.
        
        Args:
            classification: BERT classification result
            confidence: Classification confidence score
            context: Context information
            query: Original query
            
        Returns:
            Dict: Routing decision
        """
        has_vector_store = context.get('has_vector_store', True)
        has_database = context.get('has_database', False)
        
        # Determine confidence level
        confidence_level = 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
        
        if classification == "observation_query" and has_database:
            return {
                'type': QueryType.DATABASE_SEARCH,
                'classification': classification,
                'confidence': confidence_level,
                'bert_confidence': confidence,
                'parameters': {
                    'search_type': 'structured',
                    'bert_classified': True
                }
            }
        
        elif classification == "deployment_info" and has_vector_store:
            return {
                'type': QueryType.VECTOR_SEARCH,
                'classification': classification,
                'confidence': confidence_level,
                'bert_confidence': confidence,
                'parameters': {
                    'search_type': 'semantic',
                    'bert_classified': True
                }
            }
        
        elif classification == "document_search" and has_vector_store:
            return {
                'type': QueryType.VECTOR_SEARCH,
                'classification': classification,
                'confidence': confidence_level,
                'bert_confidence': confidence,
                'parameters': {
                    'search_type': 'semantic',
                    'bert_classified': True
                }
            }
        
        elif classification == "general_knowledge":
            # For general knowledge, prefer vector search if available, otherwise direct LLM
            if has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'bert_confidence': confidence,
                    'parameters': {
                        'search_type': 'semantic',
                        'bert_classified': True
                    }
                }
            else:
                return {
                    'type': QueryType.DIRECT_LLM,
                    'classification': classification,
                    'confidence': confidence_level,
                    'bert_confidence': confidence,
                    'parameters': {
                        'reason': 'General knowledge query without vector store',
                        'bert_classified': True
                    }
                }
        
        else:
            # Fallback - check what data sources are available
            if has_database and has_vector_store:
                return {
                    'type': QueryType.HYBRID_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'bert_confidence': confidence,
                    'parameters': {
                        'vector_weight': 0.5,
                        'database_weight': 0.5,
                        'reason': f'Uncertain classification: {classification}',
                        'bert_classified': True
                    }
                }
            elif has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'bert_confidence': confidence,
                    'parameters': {
                        'search_type': 'fallback',
                        'reason': f'Uncertain classification: {classification}',
                        'bert_classified': True
                    }
                }
            elif has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'bert_confidence': confidence,
                    'parameters': {
                        'search_type': 'fallback',
                        'reason': f'Uncertain classification: {classification}',
                        'bert_classified': True
                    }
                }
            else:
                return {
                    'type': QueryType.DIRECT_LLM,
                    'classification': classification,
                    'confidence': 'low',
                    'bert_confidence': confidence,
                    'parameters': {
                        'reason': f'No data sources available, classification: {classification}',
                        'bert_classified': True
                    }
                }
    
    def _llm_route_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to classify and route the query.
        
        Args:
            query: User query
            context: Additional context for routing
            
        Returns:
            Dict: Routing decision with type and parameters
        """
        has_vector_store = context.get('has_vector_store', True)
        has_database = context.get('has_database', False)
        
        # Create the classification prompt
        # Include conversation context in the prompt if available
        conversation_context = context.get('conversation_context', '')
        follow_up_info = context.get('follow_up_info', {})
        
        system_prompt = """You are a query router for an Ocean Networks Canada (ONC) oceanographic data system.

Classify queries into one of these categories:
1. **deployment_info**: Questions about sensor locations, device types, deployment setups, what sensors/devices are at specific locations
2. **observation_query**: Requests for specific sensor data with locations and/or times (temperature, salinity, pressure, etc.)
3. **general_knowledge**: Conceptual questions about oceanography, marine science, instrument explanations
4. **document_search**: Questions that need to be answered from research papers or documents

Available data sources:
- Vector database: """ + ("Available" if has_vector_store else "Not available") + """
- Ocean sensor database: """ + ("Available" if has_database else "Not available") + """

Examples:
- "What sensors are located at CBYIP?" → deployment_info
- "Give me temperature data from CBYIP on Oct 10, 2022" → observation_query  
- "What is dissolved oxygen?" → general_knowledge
- "Explain how CTD sensors work" → document_search

""" + (f"""
CONVERSATION CONTEXT:
{conversation_context}

Note: This query may be a follow-up to the previous conversation. Consider the context when classifying.
""" if conversation_context else "") + """

Respond with ONLY the category name: deployment_info, observation_query, general_knowledge, or document_search"""

        user_prompt = f"Classify this query: {query}"
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            classification = completion.choices[0].message.content.strip().lower()
            logger.debug(f"LLM classified query as: {classification}")
            
            # Map classification to routing decision
            return self._map_llm_classification_to_route(classification, context, query)
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            raise
    
    def _map_llm_classification_to_route(self, classification: str, context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Map LLM classification to routing decision.
        
        Args:
            classification: LLM classification result
            context: Context information
            query: Original query for confidence scoring
            
        Returns:
            Dict: Routing decision
        """
        has_vector_store = context.get('has_vector_store', True)
        has_database = context.get('has_database', False)
        
        if classification == "observation_query" and has_database:
            return {
                'type': QueryType.DATABASE_SEARCH,
                'classification': classification,
                'confidence': 'high',
                'parameters': {
                    'search_type': 'structured',
                    'llm_classified': True
                }
            }
        
        elif classification == "deployment_info" and has_vector_store:
            return {
                'type': QueryType.VECTOR_SEARCH,
                'classification': classification,
                'confidence': 'high',
                'parameters': {
                    'search_type': 'semantic',
                    'llm_classified': True
                }
            }
        
        elif classification == "document_search" and has_vector_store:
            return {
                'type': QueryType.VECTOR_SEARCH,
                'classification': classification,
                'confidence': 'high',
                'parameters': {
                    'search_type': 'semantic',
                    'llm_classified': True
                }
            }
        
        elif classification == "general_knowledge":
            # For general knowledge, prefer vector search if available, otherwise direct LLM
            if has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': 'medium',
                    'parameters': {
                        'search_type': 'semantic',
                        'llm_classified': True
                    }
                }
            else:
                return {
                    'type': QueryType.DIRECT_LLM,
                    'classification': classification,
                    'confidence': 'medium',
                    'parameters': {
                        'reason': 'General knowledge query without vector store',
                        'llm_classified': True
                    }
                }
        
        else:
            # Fallback - check what data sources are available
            if has_database and has_vector_store:
                return {
                    'type': QueryType.HYBRID_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'parameters': {
                        'vector_weight': 0.5,
                        'database_weight': 0.5,
                        'reason': f'Uncertain classification: {classification}',
                        'llm_classified': True
                    }
                }
            elif has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'parameters': {
                        'search_type': 'fallback',
                        'reason': f'Uncertain classification: {classification}',
                        'llm_classified': True
                    }
                }
            elif has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'parameters': {
                        'search_type': 'fallback',
                        'reason': f'Uncertain classification: {classification}',
                        'llm_classified': True
                    }
                }
            else:
                return {
                    'type': QueryType.DIRECT_LLM,
                    'classification': classification,
                    'confidence': 'low',
                    'parameters': {
                        'reason': f'No data sources available, classification: {classification}',
                        'llm_classified': True
                    }
                }
    
    def _enhance_routing_with_context(self, routing_decision: Dict[str, Any], 
                                    query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance routing decision with conversation context.
        
        Args:
            routing_decision: Original routing decision
            query: Current query
            context: Context including conversation history
            
        Returns:
            Enhanced routing decision
        """
        follow_up_info = context.get('follow_up_info', {})
        conversation_context = context.get('conversation_context', '')
        
        # Add conversation context to parameters
        if 'parameters' not in routing_decision:
            routing_decision['parameters'] = {}
        
        routing_decision['parameters']['has_conversation_context'] = bool(conversation_context)
        routing_decision['parameters']['is_follow_up'] = follow_up_info.get('is_follow_up', False)
        routing_decision['parameters']['follow_up_confidence'] = follow_up_info.get('confidence', 0.0)
        
        # If it's a follow-up question, check if we should maintain the same route type
        if follow_up_info.get('is_follow_up') and follow_up_info.get('confidence', 0) > 0.7:
            last_query_metadata = follow_up_info.get('context_info', {}).get('last_metadata', {})
            
            # If last query had similar classification, maintain consistency
            if 'classification' in last_query_metadata:
                last_classification = last_query_metadata['classification']
                current_classification = routing_decision.get('classification', '')
                
                # For high-confidence follow-ups, bias towards same route type if appropriate
                if current_classification != last_classification and follow_up_info.get('confidence', 0) > 0.8:
                    routing_decision['parameters']['route_adjustment'] = 'follow_up_bias'
                    routing_decision['parameters']['original_classification'] = current_classification
                    routing_decision['parameters']['inherited_classification'] = last_classification
        
        return routing_decision
    
    def _adjust_scores_for_follow_up(self, vector_score: float, database_score: float, 
                                   follow_up_info: Dict[str, Any]) -> Tuple[float, float]:
        """
        Adjust routing scores based on follow-up question context.
        
        Args:
            vector_score: Original vector search score
            database_score: Original database search score
            follow_up_info: Follow-up detection information
            
        Returns:
            Adjusted (vector_score, database_score)
        """
        if not follow_up_info.get('is_follow_up'):
            return vector_score, database_score
        
        confidence = follow_up_info.get('confidence', 0.0)
        context_info = follow_up_info.get('context_info', {})
        last_metadata = context_info.get('last_metadata', {})
        
        # Boost scores based on last query's routing
        if 'route_type' in last_metadata:
            last_route = last_metadata['route_type']
            
            # Apply follow-up bias - slightly favor the same route type
            bias_strength = confidence * 0.3  # Up to 30% boost
            
            if last_route == 'vector_search':
                vector_score += bias_strength
            elif last_route == 'database_search':
                database_score += bias_strength
            elif last_route in ['hybrid_search']:
                # For hybrid, boost both slightly
                vector_score += bias_strength * 0.5
                database_score += bias_strength * 0.5
        
        return vector_score, database_score
    
    def _calculate_vector_score(self, query: str) -> float:
        """Calculate score for vector search relevance."""
        score = 0.0
        word_count = len(query.split())
        
        for keyword in self.vector_keywords:
            if keyword in query:
                score += 1.0
        
        # Normalize by query length
        return score / max(word_count, 1)
    
    def _calculate_database_score(self, query: str) -> float:
        """Calculate score for database search relevance."""
        score = 0.0
        word_count = len(query.split())
        
        for keyword in self.database_keywords:
            if keyword in query:
                score += 1.0
        
        # Normalize by query length
        return score / max(word_count, 1)
    
    def _make_routing_decision(self, vector_score: float, database_score: float,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Make the final routing decision based on scores and context."""
        
        # Check if specific data sources are available
        has_vector_store = context.get('has_vector_store', True)
        has_database = context.get('has_database', False)
        
        # Decision thresholds - lowered database threshold to prioritize it
        vector_threshold = self.config.get('vector_threshold', 0.1)
        database_threshold = self.config.get('database_threshold', 0.05)  # Lower threshold for database
        hybrid_threshold = self.config.get('hybrid_threshold', 0.15)
        
        # Prioritize database search for any data queries when available
        if has_database and database_score > 0:
            # If there's any database score and database is available, use it
            if database_score > hybrid_threshold and vector_score > vector_threshold:
                return {
                    'type': QueryType.HYBRID_SEARCH,
                    'vector_score': vector_score,
                    'database_score': database_score,
                    'parameters': {
                        'vector_weight': 0.3,
                        'database_weight': 0.7  # Favor database more
                    }
                }
            else:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'database_score': database_score,
                    'parameters': {
                        'search_type': 'structured'
                    }
                }
        # Fall back to vector search for conceptual questions
        elif vector_score > vector_threshold and has_vector_store:
            return {
                'type': QueryType.VECTOR_SEARCH,
                'vector_score': vector_score,
                'parameters': {
                    'search_type': 'semantic'
                }
            }
        else:
            return {
                'type': QueryType.DIRECT_LLM,
                'parameters': {
                    'reason': 'No suitable data source or low confidence scores'
                }
            }
    
    def add_vector_keywords(self, keywords: List[str]):
        """Add new keywords for vector search routing."""
        self.vector_keywords.extend(keywords)
        logger.info(f"Added {len(keywords)} vector search keywords")
    
    def add_database_keywords(self, keywords: List[str]):
        """Add new keywords for database search routing."""
        self.database_keywords.extend(keywords)
        logger.info(f"Added {len(keywords)} database search keywords")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing configuration."""
        return {
            'vector_keywords_count': len(self.vector_keywords),
            'database_keywords_count': len(self.database_keywords),
            'bert_routing_enabled': self.use_bert_routing,
            'bert_model_available': self.bert_model is not None,
            'llm_routing_enabled': self.use_llm_routing,
            'groq_client_available': self.groq_client is not None,
            'config': self.config
        }
    
    def set_bert_routing(self, enabled: bool):
        """Enable or disable BERT-based routing."""
        if enabled and self.bert_model is None:
            logger.warning("Cannot enable BERT routing: BERT model not available")
            return False
        
        self.use_bert_routing = enabled
        logger.info(f"BERT routing {'enabled' if enabled else 'disabled'}")
        return True
    
    def set_llm_routing(self, enabled: bool):
        """Enable or disable LLM-based routing."""
        if enabled and self.groq_client is None:
            logger.warning("Cannot enable LLM routing: Groq client not available")
            return False
        
        self.use_llm_routing = enabled
        logger.info(f"LLM routing {'enabled' if enabled else 'disabled'}")
        return True