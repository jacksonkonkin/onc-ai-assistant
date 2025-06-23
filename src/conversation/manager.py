"""
Conversation context and memory management
Team: Backend + LLM team
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """Individual message in conversation."""
    type: MessageType
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary."""
        return cls(
            type=MessageType(data['type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata')
        )

class ConversationManager:
    """Manages conversation context and memory."""
    
    def __init__(self, max_history_length: int = 10, context_window_minutes: int = 30):
        """
        Initialize conversation manager.
        
        Args:
            max_history_length: Maximum number of message pairs to keep
            context_window_minutes: Minutes after which old context becomes less relevant
        """
        self.max_history_length = max_history_length
        self.context_window_minutes = context_window_minutes
        self.conversation_history: List[ConversationMessage] = []
        self.session_start_time = datetime.now()
        self.last_query_metadata = {}
        
        logger.info(f"ConversationManager initialized with max_history={max_history_length}, context_window={context_window_minutes}min")
    
    def add_user_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a user message to conversation history.
        
        Args:
            message: User's message
            metadata: Additional metadata (routing info, etc.)
        """
        msg = ConversationMessage(
            type=MessageType.USER,
            content=message,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.conversation_history.append(msg)
        self._trim_history()
        logger.debug(f"Added user message: {message[:50]}...")
    
    def add_assistant_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an assistant message to conversation history.
        
        Args:
            message: Assistant's response
            metadata: Additional metadata (sources, routing type, etc.)
        """
        msg = ConversationMessage(
            type=MessageType.ASSISTANT,
            content=message,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.conversation_history.append(msg)
        self._trim_history()
        logger.debug(f"Added assistant message: {message[:50]}...")
    
    def get_conversation_context(self, include_metadata: bool = False) -> str:
        """
        Get formatted conversation context for prompt inclusion.
        
        Args:
            include_metadata: Whether to include metadata in context
            
        Returns:
            Formatted conversation context string
        """
        if not self.conversation_history:
            return ""
        
        context_parts = ["Previous conversation context:"]
        
        # Get recent relevant messages
        relevant_messages = self._get_relevant_messages()
        
        for msg in relevant_messages:
            role = "User" if msg.type == MessageType.USER else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
            
            if include_metadata and msg.metadata:
                # Include relevant metadata like data sources, locations, etc.
                relevant_meta = self._extract_relevant_metadata(msg.metadata)
                if relevant_meta:
                    context_parts.append(f"  [Context: {relevant_meta}]")
        
        return "\n".join(context_parts)
    
    def get_last_query_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last user query."""
        for msg in reversed(self.conversation_history):
            if msg.type == MessageType.USER:
                return {
                    'query': msg.content,
                    'timestamp': msg.timestamp,
                    'metadata': msg.metadata or {}
                }
        return None
    
    def detect_follow_up_question(self, current_query: str) -> Dict[str, Any]:
        """
        Detect if current query is a follow-up to previous conversation.
        
        Args:
            current_query: Current user query
            
        Returns:
            Dict with follow-up detection results
        """
        follow_up_indicators = [
            # Pronouns and references
            "it", "that", "this", "them", "those", "there",
            # Follow-up question words
            "also", "more", "again", "further", "additional",
            # Temporal references
            "when", "before", "after", "since", "during",
            # Comparative
            "compared to", "vs", "versus", "difference",
            # Continuation
            "what about", "how about", "what if"
        ]
        
        query_lower = current_query.lower()
        
        # Check for follow-up indicators
        has_indicators = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Check for short queries (likely follow-ups)
        is_short = len(current_query.split()) <= 5
        
        # Check for recent context
        has_recent_context = len(self.conversation_history) > 0
        
        # Get potential context from last exchange
        context_info = {}
        if has_recent_context:
            last_info = self.get_last_query_info()
            if last_info:
                context_info = {
                    'last_query': last_info['query'],
                    'time_since_last': (datetime.now() - last_info['timestamp']).total_seconds() / 60,
                    'last_metadata': last_info['metadata']
                }
        
        is_follow_up = has_indicators and has_recent_context
        
        return {
            'is_follow_up': is_follow_up,
            'confidence': self._calculate_follow_up_confidence(has_indicators, is_short, has_recent_context),
            'indicators_found': [ind for ind in follow_up_indicators if ind in query_lower],
            'context_info': context_info
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        return {
            'message_count': len(self.conversation_history),
            'session_duration_minutes': (datetime.now() - self.session_start_time).total_seconds() / 60,
            'last_activity': self.conversation_history[-1].timestamp if self.conversation_history else None,
            'topics_discussed': self._extract_topics(),
            'data_queries_made': self._count_data_queries()
        }
    
    def clear_conversation(self) -> None:
        """Clear conversation history and start fresh."""
        self.conversation_history.clear()
        self.session_start_time = datetime.now()
        logger.info("Conversation history cleared")
    
    def save_conversation(self, filepath: str) -> bool:
        """
        Save conversation to file.
        
        Args:
            filepath: Path to save conversation
            
        Returns:
            Success status
        """
        try:
            conversation_data = {
                'session_start': self.session_start_time.isoformat(),
                'messages': [msg.to_dict() for msg in self.conversation_history],
                'metadata': {
                    'max_history_length': self.max_history_length,
                    'context_window_minutes': self.context_window_minutes
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            logger.info(f"Conversation saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False
    
    def load_conversation(self, filepath: str) -> bool:
        """
        Load conversation from file.
        
        Args:
            filepath: Path to load conversation from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
            
            self.session_start_time = datetime.fromisoformat(conversation_data['session_start'])
            self.conversation_history = [
                ConversationMessage.from_dict(msg_data) 
                for msg_data in conversation_data['messages']
            ]
            
            # Update settings if provided
            metadata = conversation_data.get('metadata', {})
            self.max_history_length = metadata.get('max_history_length', self.max_history_length)
            self.context_window_minutes = metadata.get('context_window_minutes', self.context_window_minutes)
            
            logger.info(f"Conversation loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False
    
    def _trim_history(self) -> None:
        """Trim conversation history to max length."""
        if len(self.conversation_history) > self.max_history_length * 2:  # 2 messages per exchange
            # Keep recent messages, preserve user-assistant pairs
            messages_to_keep = self.max_history_length * 2
            self.conversation_history = self.conversation_history[-messages_to_keep:]
            logger.debug(f"Trimmed conversation history to {len(self.conversation_history)} messages")
    
    def _get_relevant_messages(self) -> List[ConversationMessage]:
        """Get messages relevant to current context window."""
        cutoff_time = datetime.now() - timedelta(minutes=self.context_window_minutes)
        
        # Get recent messages within context window
        relevant = [
            msg for msg in self.conversation_history 
            if msg.timestamp >= cutoff_time
        ]
        
        # If no recent messages, get last few exchanges
        if not relevant and self.conversation_history:
            relevant = self.conversation_history[-6:]  # Last 3 exchanges
        
        return relevant
    
    def _extract_relevant_metadata(self, metadata: Dict[str, Any]) -> str:
        """Extract relevant metadata for context."""
        relevant_parts = []
        
        # Extract key information
        if 'classification' in metadata:
            relevant_parts.append(f"Query type: {metadata['classification']}")
        
        if 'location' in metadata:
            relevant_parts.append(f"Location: {metadata['location']}")
        
        if 'data_source' in metadata:
            relevant_parts.append(f"Data source: {metadata['data_source']}")
        
        return ", ".join(relevant_parts)
    
    def _calculate_follow_up_confidence(self, has_indicators: bool, is_short: bool, has_recent_context: bool) -> float:
        """Calculate confidence score for follow-up detection."""
        confidence = 0.0
        
        if has_indicators:
            confidence += 0.5
        if is_short:
            confidence += 0.2
        if has_recent_context:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _extract_topics(self) -> List[str]:
        """Extract topics discussed in conversation."""
        topics = []
        
        # Simple keyword extraction from user messages
        keywords = [
            'temperature', 'salinity', 'pressure', 'depth', 'oxygen', 'ph',
            'cambridge bay', 'arctic', 'ocean', 'sensor', 'ctd', 'hydrophone',
            'data', 'measurement', 'deployment', 'instrument'
        ]
        
        for msg in self.conversation_history:
            if msg.type == MessageType.USER:
                content_lower = msg.content.lower()
                for keyword in keywords:
                    if keyword in content_lower and keyword not in topics:
                        topics.append(keyword)
        
        return topics
    
    def _count_data_queries(self) -> int:
        """Count number of data-related queries."""
        data_keywords = ['data', 'measurement', 'reading', 'value', 'sensor', 'temperature', 'salinity']
        
        count = 0
        for msg in self.conversation_history:
            if msg.type == MessageType.USER:
                content_lower = msg.content.lower()
                if any(keyword in content_lower for keyword in data_keywords):
                    count += 1
        
        return count