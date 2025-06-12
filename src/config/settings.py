"""
Configuration management for ONC AI Assistant
Handles loading and validation of configuration settings
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        if config_path is None:
            config_path = "onc_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_environment()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _validate_environment(self):
        """Validate required environment variables."""
        required_keys = []
        
        # Check LLM API keys
        llm_provider = self.get('llm.provider')
        if llm_provider == 'groq':
            required_keys.append(self.get('llm.groq.api_key_env'))
        
        # Check embeddings API key

        ##embeddings_key = self.get('embeddings.api_key_env')
        ##if embeddings_key:
           ## required_keys.append(embeddings_key)
        
        missing_keys = []
        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {missing_keys}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.get('llm', {})
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self.get('embeddings', {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self.get('vector_store', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration."""
        return self.get('processing', {})
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return self.get('retrieval', {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self._config.copy()