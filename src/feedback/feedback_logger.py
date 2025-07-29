"""
Simple feedback logging system for user ratings.
Logs thumbs up/down ratings with query and response context.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackRating(Enum):
    """Feedback rating types."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class SimpleFeedbackLogger:
    """
    Simple feedback logging system that records user ratings.
    """
    
    def __init__(self, log_file_path: Optional[str] = None):
        """
        Initialize feedback logger.
        
        Args:
            log_file_path: Path to feedback log file (defaults to logs/feedback.jsonl)
        """
        if log_file_path is None:
            # Default to logs directory in project root
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file_path = str(log_dir / "feedback.jsonl")
        
        self.log_file_path = Path(log_file_path)
        
        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Feedback logger initialized with log file: {self.log_file_path}")
    
    def log_feedback(self, 
                    rating: FeedbackRating,
                    query: str,
                    response: str,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log user feedback rating.
        
        Args:
            rating: User's thumbs up/down rating
            query: Original user query
            response: System response that was rated
            metadata: Additional context (route type, confidence, etc.)
            
        Returns:
            bool: True if logging successful
        """
        try:
            feedback_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rating": rating.value,
                "query": query,
                "response": response,
                "response_length": len(response),
                "query_length": len(query),
                "metadata": metadata or {}
            }
            
            # Append to JSONL file (one JSON object per line)
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Logged {rating.value} feedback for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about logged feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        try:
            if not self.log_file_path.exists():
                return {"total_feedback": 0, "thumbs_up": 0, "thumbs_down": 0}
            
            stats = {
                "total_feedback": 0,
                "thumbs_up": 0,
                "thumbs_down": 0,
                "avg_query_length": 0,
                "avg_response_length": 0
            }
            
            total_query_length = 0
            total_response_length = 0
            
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        stats["total_feedback"] += 1
                        
                        if entry["rating"] == FeedbackRating.THUMBS_UP.value:
                            stats["thumbs_up"] += 1
                        elif entry["rating"] == FeedbackRating.THUMBS_DOWN.value:
                            stats["thumbs_down"] += 1
                        
                        total_query_length += entry.get("query_length", 0)
                        total_response_length += entry.get("response_length", 0)
            
            if stats["total_feedback"] > 0:
                stats["avg_query_length"] = round(total_query_length / stats["total_feedback"], 1)
                stats["avg_response_length"] = round(total_response_length / stats["total_feedback"], 1)
                stats["thumbs_up_percentage"] = round((stats["thumbs_up"] / stats["total_feedback"]) * 100, 1)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {"error": str(e)}
    
    def generate_simple_feedback_prompt(self, response_id: Optional[str] = None) -> str:
        """
        Generate a simple thumbs up/down feedback prompt.
        
        Args:
            response_id: Optional response ID for tracking
            
        Returns:
            Simple feedback prompt string
        """
        return "\n\n👍 👎 Was this helpful?"
    
    def format_feedback_for_display(self, rating: FeedbackRating) -> str:
        """
        Format feedback confirmation message.
        
        Args:
            rating: The rating that was given
            
        Returns:
            Confirmation message
        """
        if rating == FeedbackRating.THUMBS_UP:
            return "Thanks for the positive feedback! 👍"
        else:
            return "Thanks for the feedback. We'll work to improve! 👎"


def create_feedback_logger(config: Optional[Dict[str, Any]] = None) -> SimpleFeedbackLogger:
    """
    Factory function to create feedback logger.
    
    Args:
        config: Optional configuration for feedback logger
        
    Returns:
        SimpleFeedbackLogger instance
    """
    if config is None:
        config = {}
    
    log_file_path = config.get('log_file_path')
    return SimpleFeedbackLogger(log_file_path)