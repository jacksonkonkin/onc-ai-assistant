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
import pandas as pd
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    VECTOR_SEARCH = "vector_search"
    DATABASE_SEARCH = "database_search"
    HYBRID_SEARCH = "hybrid_search"
    DIRECT_LLM = "direct_llm"
    DATA_DOWNLOAD = "data_download"  # New type for data download queries


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
        
        # Initialize Sprint 3 Sentence Transformer classifier as fallback
        self.sentence_model = None
        self.sprint3_questions = None
        self.sprint3_labels = None
        self.sprint3_embeddings = None
        
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
        
        # Initialize Sprint 3 Sentence Transformer classifier
        try:
            self._initialize_sprint3_classifier()
            logger.info("Sprint 3 Sentence Transformer classifier enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Sprint 3 classifier: {e}")
        
        # Initialize LLM client for fallback routing
        self.groq_client = None
        if os.getenv('GROQ_API_KEY'):
            try:
                self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                logger.info("LLM fallback routing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
        
        # Enable/disable routing methods (order of preference: BERT > Sprint3 > LLM > Keyword)
        # Disable BERT routing to prioritize Sprint 3 classifier
        self.use_bert_routing = self.config.get('use_bert_routing', False) and self.bert_model is not None
        self.use_sprint3_routing = self.config.get('use_sprint3_routing', True) and self.sentence_model is not None
        self.use_llm_routing = self.config.get('use_llm_routing', True) and self.groq_client is not None
        
        # Debug logging for classifier priority
        logger.info(f"🔧 Query Router Configuration:")
        logger.info(f"   BERT routing: {'ENABLED' if self.use_bert_routing else 'DISABLED'}")
        logger.info(f"   Sprint3 routing: {'ENABLED' if self.use_sprint3_routing else 'DISABLED'}")
        logger.info(f"   LLM routing: {'ENABLED' if self.use_llm_routing else 'DISABLED'}")
        logger.info(f"   Primary classifier: {'Sprint 3 Sentence Transformer' if self.use_sprint3_routing else 'LLM' if self.use_llm_routing else 'Keyword-based'}")
    
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
            "time series", "when", "since", "from", "to", "between", "at", "pm", "am", 
            "o'clock", "hour", "minute", "morning", "afternoon", "evening", "night",
            # Request patterns
            "get", "show", "find", "retrieve", "what is the", "how much",
            "give me", "tell me"
        ]
        return self.config.get('database_keywords', default_keywords)
    
    def _initialize_sprint3_classifier(self):
        """Initialize Sprint 3 Sentence Transformer classifier."""
        # Load Sprint 3 training data from CSV files
        csv_files = [
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/deployments.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/device_info.csv", 
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/property.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/device_category.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/locations.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/deployments_2.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/device_info_2.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/property_2.csv", 
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/device_category_2.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/locations_2.csv",
            "Data-Engineering/Sprint_3/Query_Router/Simple_Python_File_Method/data_and_knowledge_queries.csv"
        ]
        
        # Try to load CSV files from current directory or absolute paths
        dfs = []
        for csv_file in csv_files:
            try:
                # Try relative path first, then absolute
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                else:
                    # Try from project root
                    root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), csv_file)
                    if os.path.exists(root_path):
                        df = pd.read_csv(root_path)  
                        dfs.append(df)
            except Exception as e:
                logger.debug(f"Could not load Sprint 3 CSV file {csv_file}: {e}")
        
        if not dfs:
            raise Exception("No Sprint 3 training data found")
        
        # Combine all dataframes
        df_all = pd.concat(dfs, ignore_index=True)
        self.sprint3_questions = df_all["question"].tolist()
        self.sprint3_labels = df_all["label"].tolist()
        
        # Load sentence transformer model
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Pre-compute embeddings for Sprint 3 questions
        self.sprint3_embeddings = self.sentence_model.encode(
            self.sprint3_questions, 
            normalize_embeddings=True
        )
        self.sprint3_embeddings = torch.tensor(self.sprint3_embeddings)
        
        logger.info(f"Loaded {len(self.sprint3_questions)} Sprint 3 training examples")
    
    def _sprint3_classify_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Classify query using Sprint 3 Sentence Transformer approach.
        
        Args:
            query: User query
            top_k: Number of top matches to consider
            
        Returns:
            Classification result with confidence
        """
        if not self.sentence_model:
            raise Exception("Sprint 3 classifier not initialized")
        
        # Encode the query
        query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)
        query_tensor = torch.tensor(query_embedding)
        
        # Calculate similarities
        similarities = util.cos_sim(query_tensor, self.sprint3_embeddings)[0]
        
        # Get top matches
        top_results = similarities.topk(top_k)
        top_indices = top_results.indices
        top_scores = top_results.values
        
        # Get the top classification
        best_index = top_indices[0].item()
        best_score = top_scores[0].item()
        classification = self.sprint3_labels[best_index]
        
        # Calculate confidence based on score difference and absolute score
        confidence = float(best_score)
        if len(top_scores) > 1:
            score_gap = float(top_scores[0] - top_scores[1])
            confidence = (confidence + score_gap) / 2  # Balance absolute score and gap
        
        return {
            'classification': classification,
            'confidence': confidence,
            'top_matches': [
                {
                    'question': self.sprint3_questions[idx.item()],
                    'label': self.sprint3_labels[idx.item()],
                    'similarity': float(score)
                }
                for idx, score in zip(top_indices, top_scores)
            ]
        }
    
    def _detect_greeting_or_social(self, query: str) -> bool:
        """
        Detect greetings and social queries that should bypass classification.
        
        Args:
            query: User query string
            
        Returns:
            True if query is a greeting/social interaction, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Exact matches for common greetings and social interactions
        exact_greetings = {
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'how are you',
            "what's up", "how's it going", 'nice to meet you', 'pleased to meet you',
            'good day', 'good night', 'howdy', 'greetings', 'salutations'
        }
        
        # Check for exact matches
        if query_lower in exact_greetings:
            return True
        
        # Check for greeting patterns in short queries (≤4 words)
        words = query_lower.split()
        if len(words) <= 4:
            greeting_words = {'hello', 'hi', 'hey', 'thanks', 'thank', 'bye', 'goodbye', 'morning', 'afternoon', 'evening'}
            if any(word in greeting_words for word in words):
                return True
        
        # Very short non-technical queries (1-2 words)
        if len(words) <= 2:
            # Technical keywords that indicate oceanographic queries
            tech_keywords = {
                'temperature', 'data', 'sensor', 'pressure', 'oxygen', 'depth', 'salinity',
                'conductivity', 'ph', 'turbidity', 'chlorophyll', 'fluorescence', 'density',
                'ctd', 'hydrophone', 'adcp', 'cambridge', 'bay', 'cby', 'onc', 'ocean',
                'marine', 'underwater', 'deployment', 'device', 'instrument', 'measurement'
            }
            
            # If no technical keywords in very short query, likely greeting/social
            if not any(tech_word in query_lower for tech_word in tech_keywords):
                return True
        
        # Check for common social question patterns
        social_patterns = [
            'how are you', 'what\'s up', 'how\'s it going', 'how do you do',
            'what time is it', 'what\'s the weather', 'tell me a joke',
            'who are you', 'what are you', 'how old are you'
        ]
        
        if any(pattern in query_lower for pattern in social_patterns):
            return True
        
        return False

    def _log_routing_decision(self, query: str, routing_decision: Dict[str, Any], method: str) -> None:
        """
        Log detailed routing decision for debugging and monitoring.
        
        Args:
            query: Original user query
            routing_decision: The routing decision made
            method: Classification method used (e.g., "Greeting Filter", "Sprint 3", "LLM")
        """
        route_type = routing_decision['type'].value if hasattr(routing_decision['type'], 'value') else str(routing_decision['type'])
        classification = routing_decision.get('classification', 'N/A')
        confidence = routing_decision.get('confidence', 'N/A')
        
        logger.info(f"🎯 {method} Routing Decision:")
        logger.info(f"   Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
        logger.info(f"   📍 Route: {route_type.upper()}")
        logger.info(f"   🏷️  Classification: {classification}")
        logger.info(f"   📊 Confidence: {confidence}")
        
        # Add specific confidence scores if available
        if 'sprint3_confidence' in routing_decision:
            logger.info(f"   🎯 Sprint 3 confidence: {routing_decision['sprint3_confidence']:.3f}")
        if 'bert_confidence' in routing_decision:
            logger.info(f"   🧠 BERT confidence: {routing_decision['bert_confidence']:.3f}")
        
        # Add parameters info
        params = routing_decision.get('parameters', {})
        if params:
            key_params = {k: v for k, v in params.items() if k in ['reason', 'search_type', 'download_type', 'fallback_applied']}
            if key_params:
                logger.info(f"   ⚙️  Key parameters: {key_params}")

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
        
        # PRE-ROUTING FILTER: Handle greetings and social queries first
        if self._detect_greeting_or_social(query):
            routing_decision = {
                'type': QueryType.DIRECT_LLM,
                'classification': 'social_greeting',
                'confidence': 'high',
                'parameters': {
                    'reason': 'Greeting or social query detected',
                    'bypass_classification': True,
                    'greeting_detected': True
                }
            }
            self._log_routing_decision(query, routing_decision, "Greeting Filter")
            return routing_decision
        
        # Check for conversation context and follow-up detection
        conversation_context = context.get('conversation_context', '')
        follow_up_info = context.get('follow_up_info', {})
        
        # Use BERT-based routing if available, otherwise fall back to Sprint 3, then LLM, then keyword-based
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
                logger.warning(f"BERT routing failed, falling back to Sprint 3: {e}")
        
        # Fallback to Sprint 3 Sentence Transformer routing
        if self.use_sprint3_routing:
            try:
                routing_decision = self._sprint3_route_query(query, context)
                
                # Enhance routing decision with conversation context
                if conversation_context or follow_up_info.get('is_follow_up'):
                    routing_decision = self._enhance_routing_with_context(
                        routing_decision, query, context
                    )
                
                self._log_routing_decision(query, routing_decision, "Sprint 3")
                return routing_decision
            except Exception as e:
                logger.warning(f"Sprint 3 routing failed, falling back to LLM: {e}")
        
        # Fallback to LLM-based routing if Sprint 3 fails
        if self.use_llm_routing:
            try:
                routing_decision = self._llm_route_query(query, context)
                
                # Enhance routing decision with conversation context
                if conversation_context or follow_up_info.get('is_follow_up'):
                    routing_decision = self._enhance_routing_with_context(
                        routing_decision, query, context
                    )
                
                self._log_routing_decision(query, routing_decision, "LLM")
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
        
        self._log_routing_decision(query, routing_decision, "Keyword-based")
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

    def _is_device_discovery_query(self, query: str) -> bool:
        """
        Check if a deployment_info query is specifically about device discovery
        
        Args:
            query: User query string
            
        Returns:
            True if query is about device discovery, False otherwise
        """
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Device discovery indicators
        device_terms = [
            'device', 'sensor', 'instrument', 'equipment',
            'ctd', 'hydrophone', 'oxygen sensor', 'ph sensor',
            'weather station', 'ice profiler', 'camera', 'fluorometer'
        ]
        
        device_discovery_patterns = [
            'what devices', 'what sensors', 'what instruments',
            'show me devices', 'show me sensors', 'show me instruments',
            'find devices', 'find sensors', 'find instruments',
            'list devices', 'list sensors', 'list instruments',
            'hydrophone devices', 'ctd sensors', 'oxygen sensors',
            'devices are at', 'sensors are at', 'instruments are at',
            'devices at cambridge bay', 'sensors at cambridge bay'
        ]
        
        # Check for device discovery patterns
        if any(pattern in query_lower for pattern in device_discovery_patterns):
            return True
        
        # Check for device terms combined with location/deployment context
        has_device_term = any(term in query_lower for term in device_terms)
        has_location_context = any(term in query_lower for term in [
            'cambridge bay', 'cambridge', 'at', 'deployed', 'location'
        ])
        
        # If query has both device terms and location context, likely device discovery
        if has_device_term and has_location_context:
            return True
        
        return False
    
    def _is_data_products_query(self, query: str) -> bool:
        """
        Check if a query is about data products discovery or download
        
        Args:
            query: User query string
            
        Returns:
            True if query is about data products, False otherwise
        """
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Data products discovery patterns
        data_products_patterns = [
            'data products', 'available data products', 'what data products',
            'data product', 'download data', 'data download', 'get data',
            'data files', 'download files', 'csv data', 'data export',
            'export data', 'data formats', 'file formats'
        ]
        
        # Check for data products patterns
        if any(pattern in query_lower for pattern in data_products_patterns):
            return True
        
        # Check for download-related terms with data context
        download_terms = ['download', 'export', 'get', 'file', 'csv', 'format']
        data_terms = ['data', 'measurements', 'readings', 'values']
        
        has_download_term = any(term in query_lower for term in download_terms)
        has_data_term = any(term in query_lower for term in data_terms)
        
        # If query has both download and data terms, likely data products
        if has_download_term and has_data_term:
            return True
        
        return False
    
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
            # Check if this is actually a data products query
            if self._is_data_products_query(query):
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'bert_confidence': confidence,
                    'parameters': {
                        'search_type': 'data_products',
                        'bert_classified': True
                    }
                }
            else:
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
        
        elif classification == "deployment_info":
            # Check if this is a data products query that should go to database
            if self._is_data_products_query(query) and has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'bert_confidence': confidence,
                    'parameters': {
                        'search_type': 'data_products',
                        'bert_classified': True
                    }
                }
            # Check if this is a device discovery query that should go to database
            elif self._is_device_discovery_query(query) and has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'bert_confidence': confidence,
                    'parameters': {
                        'search_type': 'device_discovery',
                        'bert_classified': True
                    }
                }
            elif has_vector_store:
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
            elif has_database:
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
    
    def _sprint3_route_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Sprint 3 Sentence Transformer to classify and route the query.
        
        Args:
            query: User query
            context: Additional context for routing
            
        Returns:
            Dict: Routing decision with type and parameters
        """
        # Classify using Sprint 3 approach
        sprint3_result = self._sprint3_classify_query(query)
        classification = sprint3_result['classification']
        confidence = sprint3_result['confidence']
        
        logger.debug(f"Sprint 3 classified query as: {classification} (confidence: {confidence:.3f})")
        
        # CONFIDENCE THRESHOLD CHECK: Reject low-confidence classifications
        MIN_CONFIDENCE_THRESHOLD = self.config.get('sprint3_min_confidence', 0.25)
        
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.warning(f"⚠️  Low Sprint 3 confidence ({confidence:.3f}) for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            logger.info(f"   Original classification: {classification}")
            logger.info(f"   Top matches: {[match['label'] for match in sprint3_result.get('top_matches', [])[:3]]}")
            
            # Fall back to Direct LLM for low-confidence cases
            return {
                'type': QueryType.DIRECT_LLM,
                'classification': 'low_confidence_fallback',
                'confidence': 'low',
                'sprint3_confidence': confidence,
                'parameters': {
                    'reason': f'Sprint 3 confidence ({confidence:.3f}) below threshold ({MIN_CONFIDENCE_THRESHOLD})',
                    'original_classification': classification,
                    'top_matches': sprint3_result.get('top_matches', [])[:3],
                    'sprint3_classified': True,
                    'fallback_applied': True
                }
            }
        
        # Map Sprint 3 classification to routing decision if confidence is acceptable
        return self._map_sprint3_classification_to_route(classification, confidence, context, query)
    
    def _map_sprint3_classification_to_route(self, classification: str, confidence: float, 
                                           context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Map Sprint 3 classification to routing decision.
        
        Args:
            classification: Sprint 3 classification result
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
        
        # Map Sprint 3 categories to route types
        if classification == "data_download_instance" or classification == "data_download_interval":
            # These are data requests - first show data, then offer download
            if has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,  # Changed from DATA_DOWNLOAD to DATABASE_SEARCH
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'download_type': 'instance' if classification == "data_download_instance" else 'interval',
                        'search_type': 'structured',
                        'show_data_first': True,  # New flag to indicate data preview flow
                        'sprint3_classified': True
                    }
                }
            else:
                # Fallback to direct LLM if no database
                return {
                    'type': QueryType.DIRECT_LLM,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'reason': 'Data download request but no database available',
                        'sprint3_classified': True
                    }
                }
        
        elif classification == "data_discovery":
            # Device/sensor discovery queries
            if has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'search_type': 'device_discovery',
                        'sprint3_classified': True
                    }
                }
            elif has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'search_type': 'semantic',
                        'sprint3_classified': True
                    }
                }
        
        elif classification == "general_knowledge":
            # General knowledge queries
            if has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'search_type': 'semantic',
                        'sprint3_classified': True
                    }
                }
            else:
                return {
                    'type': QueryType.DIRECT_LLM,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'reason': 'General knowledge query without vector store',
                        'sprint3_classified': True
                    }
                }
        
        elif classification == "data_and_knowledge_query":
            # Complex queries requiring both data and knowledge
            if has_database and has_vector_store:
                return {
                    'type': QueryType.HYBRID_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'vector_weight': 0.4,
                        'database_weight': 0.6,
                        'sprint3_classified': True
                    }
                }
            elif has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'search_type': 'structured',
                        'sprint3_classified': True
                    }
                }
            elif has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': confidence_level,
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'search_type': 'semantic',
                        'sprint3_classified': True
                    }
                }
        
        # Fallback for unknown classifications
        else:
            if has_database and has_vector_store:
                return {
                    'type': QueryType.HYBRID_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'vector_weight': 0.5,
                        'database_weight': 0.5,
                        'reason': f'Unknown Sprint 3 classification: {classification}',
                        'sprint3_classified': True
                    }
                }
            elif has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'search_type': 'fallback',
                        'reason': f'Unknown Sprint 3 classification: {classification}',
                        'sprint3_classified': True
                    }
                }
            elif has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': 'low',
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'search_type': 'fallback',
                        'reason': f'Unknown Sprint 3 classification: {classification}',
                        'sprint3_classified': True
                    }
                }
            else:
                return {
                    'type': QueryType.DIRECT_LLM,
                    'classification': classification,
                    'confidence': 'low',
                    'sprint3_confidence': confidence,
                    'parameters': {
                        'reason': f'No data sources available, classification: {classification}',
                        'sprint3_classified': True
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
        
        elif classification == "deployment_info":
            # Check if this is a device discovery query that should go to database
            if self._is_device_discovery_query(query) and has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': 'high',
                    'parameters': {
                        'search_type': 'device_discovery',
                        'llm_classified': True
                    }
                }
            elif has_vector_store:
                return {
                    'type': QueryType.VECTOR_SEARCH,
                    'classification': classification,
                    'confidence': 'high',
                    'parameters': {
                        'search_type': 'semantic',
                        'llm_classified': True
                    }
                }
            elif has_database:
                return {
                    'type': QueryType.DATABASE_SEARCH,
                    'classification': classification,
                    'confidence': 'high',
                    'parameters': {
                        'search_type': 'structured',
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
            
            # Apply follow-up bias - favor the same route type more strongly
            bias_strength = confidence * 0.5  # Up to 50% boost
            
            if last_route == 'vector_search':
                vector_score += bias_strength
            elif last_route == 'database_search':
                database_score += bias_strength
                # Extra boost for database search follow-ups with time references
                if any(time_word in follow_up_info.get('context_info', {}).get('last_query', '').lower() 
                       for time_word in ['time', 'data', 'temperature', 'oxygen', 'sensor']):
                    database_score += 0.3  # Additional boost for data-related follow-ups
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