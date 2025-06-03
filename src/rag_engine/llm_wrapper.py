"""
LLM wrapper and interface abstraction
Team: LLM team
"""

import os
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from groq import Groq

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass


class GroqLLMWrapper(LLMInterface):
    """Groq API wrapper implementation."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", 
                 temperature: float = 0.1, max_tokens: int = 1000):
        """Initialize Groq LLM wrapper."""
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized Groq LLM: {model}")
    
    def invoke(self, prompt: str) -> str:
        """Generate response from prompt."""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "groq",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class OpenAILLMWrapper(LLMInterface):
    """OpenAI API wrapper implementation (placeholder for future use)."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", 
                 temperature: float = 0.1, max_tokens: int = 1000):
        """Initialize OpenAI LLM wrapper."""
        # TODO: Implement OpenAI client initialization
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"OpenAI LLM wrapper initialized (placeholder): {model}")
    
    def invoke(self, prompt: str) -> str:
        """Generate response from prompt."""
        # TODO: Implement OpenAI API call
        return "OpenAI integration not yet implemented"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class LLMWrapper:
    """Factory class for creating LLM instances."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize LLM wrapper with configuration.
        
        Args:
            llm_config: LLM configuration dictionary
        """
        self.config = llm_config
        self.llm = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM based on configuration."""
        provider = self.config.get('provider', 'groq')
        
        if provider == 'groq':
            self._setup_groq()
        elif provider == 'openai':
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _setup_groq(self):
        """Setup Groq LLM."""
        groq_config = self.config.get('groq', {})
        api_key_env = groq_config.get('api_key_env', 'GROQ_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        
        self.llm = GroqLLMWrapper(
            api_key=api_key,
            model=groq_config.get('model', 'llama-3.3-70b-versatile'),
            temperature=groq_config.get('temperature', 0.1)
        )
    
    def _setup_openai(self):
        """Setup OpenAI LLM."""
        openai_config = self.config.get('openai', {})
        api_key_env = openai_config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        
        self.llm = OpenAILLMWrapper(
            api_key=api_key,
            model=openai_config.get('model', 'gpt-4'),
            temperature=openai_config.get('temperature', 0.1)
        )
    
    def invoke(self, prompt: str) -> str:
        """Generate response from prompt."""
        if not self.llm:
            raise ValueError("LLM not initialized")
        return self.llm.invoke(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.llm:
            return {"error": "LLM not initialized"}
        return self.llm.get_model_info()