"""
Query routing logic and decision making
Teams: Backend team + Data team
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

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
            "data", "measurement", "sensor", "instrument", "value",
            "temperature", "salinity", "pressure", "time series",
            "latest", "current", "recent", "when", "where",
            "station", "location", "coordinates", "depth"
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
        query_lower = query.lower()
        
        # Analyze query content
        vector_score = self._calculate_vector_score(query_lower)
        database_score = self._calculate_database_score(query_lower)
        
        # Make routing decision
        routing_decision = self._make_routing_decision(
            vector_score, database_score, context
        )
        
        logger.info(f"Routed query to: {routing_decision['type']}")
        return routing_decision
    
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
        has_database = context.get('has_database', False)  # Will be True when Ocean 3 DB is connected
        
        # Decision thresholds
        vector_threshold = self.config.get('vector_threshold', 0.1)
        database_threshold = self.config.get('database_threshold', 0.1)
        hybrid_threshold = self.config.get('hybrid_threshold', 0.15)
        
        # Routing logic
        if vector_score > hybrid_threshold and database_score > hybrid_threshold and has_database:
            return {
                'type': QueryType.HYBRID_SEARCH,
                'vector_score': vector_score,
                'database_score': database_score,
                'parameters': {
                    'vector_weight': 0.6,
                    'database_weight': 0.4
                }
            }
        elif database_score > vector_score and database_score > database_threshold and has_database:
            return {
                'type': QueryType.DATABASE_SEARCH,
                'database_score': database_score,
                'parameters': {
                    'search_type': 'structured'
                }
            }
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
            'config': self.config
        }