"""
Generic RAG (Retrieval-Augmented Generation) Application for JSON Data
====================================================================

This application provides a flexible foundation for building AI-powered chatbots
that can answer questions about data stored in JSON files. It's designed to be
easily configurable and extensible for various data structures and use cases.

Key Components:
1. ConfigManager: Handles YAML-based configuration management
2. JSONDataLoader: Processes different JSON structures into documents
3. GenericRAGApplication: Main RAG pipeline implementation

Architecture Overview:
1. Load JSON data and convert to Document objects
2. Split documents into manageable chunks
3. Create vector embeddings for semantic search
4. Setup LLM chain for generating responses
5. Retrieve relevant context and generate answers

Author: SENG 499 Project Team
Purpose: Foundation for ONC Ocean Data Assistant
"""

import os
import json
import yaml
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import argparse
import logging
from pathlib import Path

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Handles configuration loading and validation for the RAG application.
    
    This class manages YAML-based configuration files that control all aspects
    of the RAG pipeline, from data processing to LLM parameters. If no config
    file exists, it creates one with sensible defaults.
    
    Attributes:
        config_path (str): Path to the YAML configuration file
        config (Dict): Loaded configuration dictionary
    
    Example:
        >>> config_manager = ConfigManager("my_config.yaml")
        >>> chunk_size = config_manager.config['processing']['chunk_size']
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file or create default if it doesn't exist.
        
        The configuration structure includes:
        - data: Settings for JSON data processing
        - processing: Text splitting and chunking parameters
        - embeddings: Vector embedding configuration
        - llm: Language model settings
        - retrieval: Search and similarity parameters
        - categories: Content categorization options
        
        Returns:
            Dict[str, Any]: Complete configuration dictionary
            
        Raises:
            yaml.YAMLError: If the YAML file is malformed
        """
        # Default configuration with comprehensive settings
        default_config = {
            "data": {
                # Path to the JSON data file
                "file_path": "data.json",
                
                # JSON structure type: 'flat' (single object), 'nested' (objects with objects), 
                # 'array' (list of objects), or 'auto' (detect automatically)
                "structure_type": "auto",
                
                # For array structures, which field serves as the unique identifier
                "key_field": None,
                
                # Specific fields to include in content (empty = all fields)
                "content_fields": [],
                
                # Fields to exclude from content processing
                "exclude_fields": [],
                
                # Whether to flatten nested objects into readable strings
                "flatten_nested": True
            },
            "processing": {
                # Size of each text chunk in tokens (smaller = more precise, larger = more context)
                "chunk_size": 300,
                
                # Overlap between chunks to preserve context across boundaries
                "chunk_overlap": 30,
                
                # Number of documents to process at once (affects memory usage)
                "batch_size": 15,
                
                # Text separators for splitting chunks (tried in order)
                "separators": ["\n\n", "\n", ": ", " - "]
            },
            "embeddings": {
                # Embedding provider: 'openai', 'huggingface', 'sentence-transformers'
                "provider": "openai",
                
                # Specific embedding model to use
                "model": "text-embedding-ada-002",
                
                # Environment variable containing the API key
                "api_key_env": "OPENAI_API_KEY"
            },
            "llm": {
                # LLM provider: 'ollama', 'openai', 'anthropic'
                "provider": "ollama",
                
                # Specific model name (e.g., 'llama3.1', 'gpt-4', 'claude-3')
                "model": "llama3.1",
                
                # Generation temperature (0 = deterministic, 1 = creative)
                "temperature": 0
            },
            "retrieval": {
                # Number of most relevant documents to retrieve per query
                "k": 6,
                
                # Minimum similarity score to consider a document relevant
                "similarity_threshold": 0.7
            },
            "categories": {
                # Whether to enable content categorization
                "enabled": False,
                
                # Automatically group fields by type (dates, IDs, etc.)
                "auto_categorize": True,
                
                # Custom category definitions: {"category_name": ["keyword1", "keyword2"]}
                "custom_categories": {}
            }
        }
        
        # Load existing configuration or create default
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge user config with defaults (user config takes precedence)
                    default_config.update(user_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except yaml.YAMLError as e:
                logger.error(f"Error loading YAML config: {e}")
                logger.info("Using default configuration")
        else:
            # Create default config file for future customization
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default config file: {self.config_path}")
        
        return default_config


class JSONDataLoader:
    """
    Generic JSON data loader that handles various JSON structures and converts
    them into LangChain Document objects for processing.
    
    This class can automatically detect and handle three main JSON structures:
    1. Flat: Single object with key-value pairs
    2. Nested: Object containing other objects as values
    3. Array: List of objects
    
    Features:
    - Automatic structure detection
    - Field filtering (include/exclude specific fields)
    - Content categorization
    - Nested object flattening
    - Metadata preservation
    
    Attributes:
        config (Dict): Full application configuration
        data_config (Dict): Data-specific configuration subset
        categories_config (Dict): Categorization configuration subset
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the JSON data loader with configuration.
        
        Args:
            config (Dict[str, Any]): Application configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.categories_config = config.get('categories', {})
    
    def load_json_data(self, file_path: str) -> List[Document]:
        """
        Load JSON data from file and convert to Document objects.
        
        This is the main entry point for processing JSON data. It handles:
        1. Loading and parsing the JSON file
        2. Detecting or using configured structure type
        3. Converting to Document objects with metadata
        4. Applying field filtering and categorization
        
        Args:
            file_path (str): Path to the JSON data file
            
        Returns:
            List[Document]: List of LangChain Document objects
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            json.JSONDecodeError: If the JSON is malformed
            ValueError: If unsupported structure type is detected
        """
        logger.info(f"Loading data from {file_path}")
        
        # Load and parse JSON file
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise
        
        # Determine structure type automatically or use configured type
        if self.data_config['structure_type'] == 'auto':
            structure_type = self._detect_structure(data)
            logger.info(f"Auto-detected structure type: {structure_type}")
        else:
            structure_type = self.data_config['structure_type']
            logger.info(f"Using configured structure type: {structure_type}")
        
        # Process data based on its structure
        if structure_type == 'flat':
            documents = self._process_flat_structure(data, file_path)
        elif structure_type == 'nested':
            documents = self._process_nested_structure(data, file_path)
        elif structure_type == 'array':
            documents = self._process_array_structure(data, file_path)
        else:
            raise ValueError(f"Unsupported structure type: {structure_type}")
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _detect_structure(self, data: Any) -> str:
        """
        Automatically detect the structure type of JSON data.
        
        Detection logic:
        - List -> 'array'
        - Dict with nested objects -> 'nested'
        - Dict with primitive values -> 'flat'
        
        Args:
            data: Parsed JSON data
            
        Returns:
            str: Detected structure type ('flat', 'nested', or 'array')
            
        Raises:
            ValueError: If data is neither dict nor list
        """
        if isinstance(data, list):
            return 'array'
        elif isinstance(data, dict):
            # Check if any values are complex (dicts or lists)
            has_nested = any(isinstance(v, (dict, list)) for v in data.values())
            if has_nested:
                # Further check: if most values are dicts, it's likely a nested structure
                # where keys are identifiers and values are objects
                sample_values = list(data.values())[:3]  # Sample first 3 values
                if all(isinstance(v, dict) for v in sample_values):
                    return 'nested'
                else:
                    return 'nested'  # Mixed nested structure
            else:
                return 'flat'
        else:
            raise ValueError("JSON must be either an object or an array")
    
    def _process_flat_structure(self, data: Dict[str, Any], file_path: str) -> List[Document]:
        """
        Process flat JSON structure (single object with key-value pairs).
        
        Example input:
        {
            "name": "Ocean Station",
            "temperature": 15.2,
            "location": "Pacific"
        }
        
        Creates a single document with all key-value pairs as content.
        
        Args:
            data (Dict[str, Any]): Flat JSON object
            file_path (str): Source file path for metadata
            
        Returns:
            List[Document]: Single document containing all data
        """
        content_parts = []
        
        # Process each key-value pair
        for key, value in data.items():
            if self._should_include_field(key):
                content_parts.append(f"{key}: {value}")
        
        # Combine all parts into single content string
        content = "\n".join(content_parts)
        
        # Create metadata for tracking and debugging
        metadata = {
            "source": file_path,
            "document_type": "flat_json",
            "total_fields": len(data)
        }
        
        return [Document(page_content=content, metadata=metadata)]
    
    def _process_nested_structure(self, data: Dict[str, Any], file_path: str) -> List[Document]:
        """
        Process nested JSON structure (object with nested objects as values).
        
        Example input:
        {
            "station_1": {
                "name": "Pacific Station",
                "temperature": 15.2
            },
            "station_2": {
                "name": "Atlantic Station", 
                "temperature": 12.8
            }
        }
        
        Creates one document per top-level key (station_1, station_2).
        
        Args:
            data (Dict[str, Any]): Nested JSON object
            file_path (str): Source file path for metadata
            
        Returns:
            List[Document]: One document per nested object
        """
        documents = []
        
        # Process each top-level key-value pair
        for key, value in data.items():
            if isinstance(value, dict):
                # Create document for nested object
                content = self._format_nested_content(key, value)
                metadata = {
                    "source": file_path,
                    "identifier": key,
                    "document_type": "nested_json",
                    "total_fields": len(value)
                }
                documents.append(Document(page_content=content, metadata=metadata))
            elif self._should_include_field(key):
                # Handle non-dict values at top level (simple fields)
                content = f"{key}: {value}"
                metadata = {
                    "source": file_path,
                    "identifier": key,
                    "document_type": "simple_field"
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _process_array_structure(self, data: List[Dict[str, Any]], file_path: str) -> List[Document]:
        """
        Process array JSON structure (list of objects).
        
        Example input:
        [
            {
                "id": "station_1",
                "name": "Pacific Station",
                "temperature": 15.2
            },
            {
                "id": "station_2", 
                "name": "Atlantic Station",
                "temperature": 12.8
            }
        ]
        
        Creates one document per array item.
        
        Args:
            data (List[Dict[str, Any]]): List of JSON objects
            file_path (str): Source file path for metadata
            
        Returns:
            List[Document]: One document per array item
        """
        documents = []
        key_field = self.data_config.get('key_field')
        
        # Process each item in the array
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Determine identifier for this item
                # Use specified key field if available, otherwise use index
                if key_field and key_field in item:
                    identifier = str(item[key_field])
                else:
                    identifier = f"item_{i}"
                
                # Format content and create document
                content = self._format_nested_content(identifier, item)
                metadata = {
                    "source": file_path,
                    "identifier": identifier,
                    "document_type": "array_item",
                    "index": i,
                    "total_fields": len(item)
                }
                
                # Add key field value to metadata if specified
                if key_field and key_field in item:
                    metadata[f"key_{key_field}"] = item[key_field]
                
                documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _format_nested_content(self, identifier: str, data: Dict[str, Any]) -> str:
        """
        Format nested content with optional categorization.
        
        This method creates a readable text representation of a nested object,
        optionally organizing fields into categories for better structure.
        
        Args:
            identifier (str): Unique identifier for this content
            data (Dict[str, Any]): Object data to format
            
        Returns:
            str: Formatted content string
        """
        content_parts = [f"Identifier: {identifier}\n"]
        
        if self.categories_config.get('enabled', False):
            # Use categorization if enabled
            content_parts.extend(self._categorize_content(data))
        else:
            # Simple flat formatting
            content_parts.extend(self._format_flat_content(data))
        
        return "\n".join(content_parts)
    
    def _categorize_content(self, data: Dict[str, Any]) -> List[str]:
        """
        Categorize content fields based on configuration.
        
        This method groups fields into categories (e.g., "Identifiers", "Dates", etc.)
        to create more structured and readable content.
        
        Args:
            data (Dict[str, Any]): Object data to categorize
            
        Returns:
            List[str]: List of formatted category sections
        """
        categories = self.categories_config.get('custom_categories', {})
        categorized_content = []
        
        # Use auto-categorization if no custom categories defined
        if not categories and self.categories_config.get('auto_categorize', True):
            categories = self._auto_categorize_fields(data)
        
        # Categorize fields based on keyword matching
        for category, keywords in categories.items():
            category_items = []
            for field, value in data.items():
                if self._should_include_field(field):
                    # Check if field name contains any category keywords
                    if any(kw.lower() in field.lower() for kw in keywords):
                        category_items.append(f"  - {field}: {value}")
            
            # Add category section if it has items
            if category_items:
                categorized_content.append(f"\n{category}:")
                categorized_content.extend(category_items)
        
        # Handle uncategorized fields
        categorized_fields = set()
        for keywords in categories.values():
            for field in data.keys():
                if any(kw.lower() in field.lower() for kw in keywords):
                    categorized_fields.add(field)
        
        # Add remaining fields to "Other" category
        uncategorized_fields = [f"  - {k}: {v}" for k, v in data.items() 
                              if k not in categorized_fields and self._should_include_field(k)]
        
        if uncategorized_fields:
            categorized_content.append("\nOther:")
            categorized_content.extend(uncategorized_fields)
        
        return categorized_content
    
    def _auto_categorize_fields(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Automatically categorize fields based on common patterns.
        
        This creates standard categories based on field name patterns commonly
        found in data structures.
        
        Args:
            data (Dict[str, Any]): Object data (used for future enhancements)
            
        Returns:
            Dict[str, List[str]]: Category name to keywords mapping
        """
        categories = {
            "Identifiers": ["id", "name", "title", "key", "identifier"],
            "Dates & Time": ["date", "time", "created", "updated", "timestamp"],
            "Contact Info": ["email", "phone", "address", "contact"],
            "Financial": ["price", "cost", "amount", "money", "revenue", "profit"],
            "Status & State": ["status", "state", "active", "enabled", "completed"],
            "Measurements": ["count", "total", "sum", "average", "rate", "percent"]
        }
        return categories
    
    def _format_flat_content(self, data: Dict[str, Any]) -> List[str]:
        """
        Format content without categorization (simple key-value pairs).
        
        Args:
            data (Dict[str, Any]): Object data to format
            
        Returns:
            List[str]: List of formatted key-value strings
        """
        formatted = []
        for key, value in data.items():
            if self._should_include_field(key):
                if isinstance(value, (dict, list)):
                    # Handle nested structures
                    if self.data_config.get('flatten_nested', True):
                        formatted.append(f"{key}: {self._flatten_value(value)}")
                    else:
                        # Keep as JSON for complex structures
                        formatted.append(f"{key}: {json.dumps(value, indent=2)}")
                else:
                    # Simple value
                    formatted.append(f"{key}: {value}")
        return formatted
    
    def _flatten_value(self, value: Any) -> str:
        """
        Flatten nested values into a readable string representation.
        
        This method converts complex nested structures (dicts, lists) into
        human-readable strings while preserving the essential information.
        
        Args:
            value: The value to flatten (can be dict, list, or primitive)
            
        Returns:
            str: Flattened string representation
            
        Example:
            Input: {"location": {"lat": 48.5, "lng": -123.4}, "depth": 100}
            Output: "(location: (lat: 48.5, lng: -123.4), depth: 100)"
        """
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    parts.append(f"{k}: {self._flatten_value(v)}")
                else:
                    parts.append(f"{k}: {v}")
            return "(" + ", ".join(parts) + ")"
        elif isinstance(value, list):
            # Convert list items to strings and join
            return "[" + ", ".join(str(item) for item in value) + "]"
        else:
            return str(value)
    
    def _should_include_field(self, field_name: str) -> bool:
        """
        Check if a field should be included based on include/exclude configuration.
        
        Logic:
        1. If content_fields is specified, only include those fields
        2. Otherwise, include all fields except those in exclude_fields
        
        Args:
            field_name (str): Name of the field to check
            
        Returns:
            bool: True if field should be included, False otherwise
        """
        content_fields = self.data_config.get('content_fields', [])
        exclude_fields = self.data_config.get('exclude_fields', [])
        
        # If specific fields are listed, only include those
        if content_fields:
            return field_name in content_fields
        
        # Otherwise, exclude specified fields
        return field_name not in exclude_fields


class GenericRAGApplication:
    """
    Main RAG (Retrieval-Augmented Generation) application class.
    
    This class orchestrates the entire RAG pipeline:
    1. Data loading and preprocessing
    2. Text chunking and embedding
    3. Vector store creation and management
    4. LLM setup and chain creation
    5. Query processing and response generation
    
    The application is designed to be flexible and configurable through YAML files,
    making it easy to adapt for different datasets and use cases.
    
    Attributes:
        config_manager (ConfigManager): Configuration management instance
        config (Dict): Complete application configuration
        data_loader (JSONDataLoader): Data loading and processing instance
        vectorstore (SKLearnVectorStore): Vector database for similarity search
        retriever: LangChain retriever for document search
        rag_chain: Complete RAG processing chain
        documents (List[Document]): Original loaded documents
        doc_splits (List[Document]): Chunked documents ready for embedding
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAG application with configuration.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        # Initialize components
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.data_loader = JSONDataLoader(self.config)
        
        # Initialize empty attributes (will be set during setup)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.documents = None
        self.doc_splits = None
        
    def setup(self, data_file: Optional[str] = None):
        """
        Setup the complete RAG application pipeline.
        
        This method performs all necessary setup steps:
        1. Load and process JSON data
        2. Split documents into chunks
        3. Create vector embeddings
        4. Setup LLM and processing chain
        
        Args:
            data_file (Optional[str]): Path to data file (overrides config)
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If configuration is invalid
        """
        logger.info("Starting RAG application setup...")
        
        # Step 1: Load and process data
        data_file = data_file or self.config['data']['file_path']
        logger.info(f"Loading data from: {data_file}")
        self.documents = self.data_loader.load_json_data(data_file)
        
        # Step 2: Split documents into manageable chunks
        logger.info("Splitting documents into chunks...")
        self._setup_text_splitter()
        self.doc_splits = self.text_splitter.split_documents(self.documents)
        logger.info(f"Split {len(self.documents)} documents into {len(self.doc_splits)} chunks")
        
        # Step 3: Create vector store for similarity search
        logger.info("Creating vector store...")
        self._setup_vectorstore()
        
        # Step 4: Setup LLM and complete processing chain
        logger.info("Setting up LLM and processing chain...")
        self._setup_llm_chain()
        
        logger.info("RAG application setup complete!")
    
    def _setup_text_splitter(self):
        """
        Setup text splitter for dividing documents into chunks.
        
        Text splitting is crucial for RAG because:
        1. Embedding models have token limits
        2. Smaller chunks provide more precise retrieval
        3. Overlaps preserve context between chunks
        
        Uses RecursiveCharacterTextSplitter with tiktoken encoding for
        accurate token counting.
        """
        processing_config = self.config['processing']
        
        # Create text splitter with tiktoken encoder for accurate token counting
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=processing_config['chunk_size'],      # Maximum tokens per chunk
            chunk_overlap=processing_config['chunk_overlap'], # Tokens to overlap between chunks
            separators=processing_config['separators']       # Preferred split points
        )
        
        logger.info(f"Text splitter configured: {processing_config['chunk_size']} tokens per chunk, "
                   f"{processing_config['chunk_overlap']} token overlap")
    
    def _setup_vectorstore(self):
        """
        Setup vector store with embeddings for similarity search.
        
        The vector store enables semantic search by:
        1. Converting text to numerical vectors (embeddings)
        2. Storing vectors for efficient similarity search
        3. Providing retrieval interface for the RAG chain
        
        Processes documents in batches to manage memory usage.
        
        Raises:
            ValueError: If embedding provider is not supported
            RuntimeError: If API key is missing or invalid
        """
        embeddings_config = self.config['embeddings']
        
        # Create embedding function based on configured provider
        if embeddings_config['provider'] == 'openai':
            # Get API key from environment variable
            api_key = os.getenv(embeddings_config['api_key_env'])
            if not api_key:
                raise ValueError(f"Environment variable {embeddings_config['api_key_env']} not set. "
                               f"Please set your OpenAI API key.")
            
            # Initialize OpenAI embeddings
            embedding_function = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=embeddings_config.get('model', 'text-embedding-ada-002')
            )
            logger.info(f"Using OpenAI embeddings: {embeddings_config.get('model')}")
        else:
            # Future: Add support for other embedding providers
            raise ValueError(f"Unsupported embedding provider: {embeddings_config['provider']}. "
                           f"Currently supported: openai")
        
        # Create vector store by processing documents in batches
        # This prevents memory issues with large datasets
        self.vectorstore = self._create_vectorstore_in_batches(
            self.doc_splits, 
            embedding_function, 
            self.config['processing']['batch_size']
        )
        
        # Setup retriever with configuration parameters
        retrieval_config = self.config['retrieval']
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": retrieval_config['k']}  # Number of documents to retrieve
        )
        logger.info(f"Vector store created with {len(self.doc_splits)} documents, "
                   f"retrieving top {retrieval_config['k']} matches per query")
    
    def _create_vectorstore_in_batches(self, doc_splits: List[Document], 
                                     embedding_function, batch_size: int):
        """
        Create vector store by processing documents in batches.
        
        Batch processing is important for:
        1. Memory management with large datasets
        2. API rate limiting
        3. Progress tracking and error recovery
        
        Args:
            doc_splits (List[Document]): Documents to process
            embedding_function: Function to create embeddings
            batch_size (int): Number of documents per batch
            
        Returns:
            SKLearnVectorStore: Populated vector store
            
        Raises:
            Exception: If embedding creation fails for a batch
        """
        logger.info(f"Processing {len(doc_splits)} document chunks in batches of {batch_size}")
        
        # Create initial vectorstore with first batch
        first_batch = doc_splits[:batch_size]
        vectorstore = SKLearnVectorStore.from_documents(
            documents=first_batch,
            embedding=embedding_function
        )
        logger.info(f"✓ Processed batch 1/{(len(doc_splits) + batch_size - 1) // batch_size} "
                   f"({len(first_batch)} chunks)")
        
        # Process remaining batches
        for i in range(batch_size, len(doc_splits), batch_size):
            batch = doc_splits[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(doc_splits) + batch_size - 1) // batch_size
            
            try:
                # Extract text and metadata from documents
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                # Add to existing vectorstore
                vectorstore.add_texts(texts, metadatas=metadatas)
                logger.info(f"✓ Processed batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
            except Exception as e:
                logger.error(f"✗ Error processing batch {batch_num}: {e}")
                logger.warning("Continuing with next batch...")
                continue
        
        return vectorstore
    
    def _setup_llm_chain(self):
        """
        Setup Language Model and create the complete RAG processing chain.
        
        The RAG chain combines:
        1. Document retrieval (finding relevant context)
        2. Prompt formatting (structuring input for LLM)
        3. LLM inference (generating responses)
        4. Output parsing (formatting final responses)
        
        This creates a complete pipeline from question to answer.
        
        Raises:
            ValueError: If LLM provider is not supported
            ConnectionError: If LLM service is unavailable
        """
        llm_config = self.config['llm']
        
        # Create LLM instance based on configured provider
        if llm_config['provider'] == 'ollama':
            # Ollama for local LLM deployment
            self.llm = ChatOllama(
                model=llm_config['model'],
                temperature=llm_config.get('temperature', 0)
            )
            logger.info(f"Using Ollama LLM: {llm_config['model']} (temp: {llm_config.get('temperature', 0)})")
        else:
            # Future: Add support for other LLM providers (OpenAI, Anthropic, etc.)
            raise ValueError(f"Unsupported LLM provider: {llm_config['provider']}. "
                           f"Currently supported: ollama")
        
        # Create sophisticated prompt template for RAG responses
        self.prompt = PromptTemplate(
            template="""You are an expert data analyst assistant helping users understand and explore their data.

INSTRUCTIONS:
- Use ONLY the provided data to answer questions
- Be specific and cite actual values, identifiers, and details from the data
- If comparing items, highlight key differences and similarities
- If the data doesn't contain enough information to answer fully, say so
- Organize your response clearly with relevant details
- When appropriate, suggest follow-up questions or related insights

CONTEXT DATA:
{documents}

USER QUESTION: {question}

DETAILED ANSWER:""",
            input_variables=["question", "documents"],
        )
        
        # Create the complete RAG processing chain
        # Pipeline: Prompt → LLM → Output Parser
        self.rag_chain = self.prompt | self.llm | StrOutputParser()
        
        logger.info("RAG processing chain created successfully")
    
    def query(self, question: str) -> str:
        """
        Process a user question and generate an answer using RAG.
        
        The query process:
        1. Retrieve relevant documents using semantic search
        2. Format documents and question for the LLM
        3. Generate response using the RAG chain
        4. Return formatted answer
        
        Args:
            question (str): User's question about the data
            
        Returns:
            str: Generated answer based on retrieved context
            
        Raises:
            ValueError: If application hasn't been setup
            Exception: If query processing fails
            
        Example:
            >>> app = GenericRAGApplication()
            >>> app.setup("ocean_data.json")
            >>> answer = app.query("What's the temperature at Station A?")
            >>> print(answer)
        """
        # Ensure application is properly initialized
        if not self.retriever or not self.rag_chain:
            raise ValueError("Application not setup. Call setup() first.")
        
        try:
            # Step 1: Retrieve relevant documents using semantic search
            logger.info(f"Processing query: {question[:100]}{'...' if len(question) > 100 else ''}")
            documents = self.retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} relevant documents")
            
            # Step 2: Format documents for LLM context
            doc_texts = []
            for i, doc in enumerate(documents):
                # Extract identifier from metadata for reference
                identifier = doc.metadata.get('identifier', f'Document_{i+1}')
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('document_type', 'Unknown')
                
                # Format document with metadata for clarity
                doc_header = f"[{identifier}] (Source: {source}, Type: {doc_type})"
                doc_texts.append(f"{doc_header}\n{doc.page_content}")
            
            # Combine all retrieved documents
            combined_docs = "\n\n" + "="*50 + "\n\n".join(doc_texts)
            
            # Step 3: Generate answer using RAG chain
            answer = self.rag_chain.invoke({
                "question": question, 
                "documents": combined_docs
            })
            
            logger.info("Query processed successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error processing your question: {str(e)}"
    
    def interactive_mode(self):
        """
        Run interactive question-answering mode.
        
        This provides a command-line interface for users to:
        1. Ask questions about their data
        2. Get immediate responses
        3. See system statistics
        4. Exit gracefully
        
        Features:
        - Continuous questioning loop
        - Graceful error handling
        - Progress indicators
        - Session statistics
        
        Usage:
            Type questions naturally, like:
            - "What sensors are available in Cambridge Bay?"
            - "Show me temperature data from last week"
            - "Compare stations by depth"
            
        Exit with: 'quit', 'exit', 'q', or Ctrl+C
        """
        # Display welcome message and system information
        print("\n" + "="*70)
        print("GENERIC JSON DATA Q&A SYSTEM")
        print("Ask natural language questions about your data!")
        print("="*70)
        print(f"Loaded: {len(self.documents)} documents")
        print(f"Searchable chunks: {len(self.doc_splits)}")
        print(f"LLM: {self.config['llm']['provider']} - {self.config['llm']['model']}")
        print(f"Retrieval: Top {self.config['retrieval']['k']} most relevant chunks")
        print("-"*70)
        print("Example questions:")
        print("   • 'What data is available for [location]?'")
        print("   • 'Show me temperature measurements'")
        print("   • 'Compare different stations'")
        print("   • 'What sensors are deployed?'")
        print("-"*70)
        print("Type 'quit', 'exit', 'q', or press Ctrl+C to stop")
        print("="*70 + "\n")
        
        question_count = 0
        
        while True:
            try:
                # Get user input
                question = input("Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("\nThank you for using the Q&A system!")
                    if question_count > 0:
                        print(f"You asked {question_count} question{'s' if question_count != 1 else ''}.")
                    print("Goodbye!")
                    break
                
                # Process the question
                question_count += 1
                print("Analyzing data...", end='', flush=True)
                
                # Generate answer
                answer = self.query(question)
                
                # Clear loading message and display answer
                print("\r" + " "*30 + "\r", end='')
                print(f"**Answer:** {answer}\n")
                print("─" * 50)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                print("Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {str(e)}")
                print("Please try again.\n")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information for debugging and monitoring.
        
        Returns:
            Dict[str, Any]: System status and configuration information
            
        Example:
            >>> app.get_system_info()
            {
                'documents_loaded': 150,
                'chunks_created': 450,
                'embedding_model': 'text-embedding-ada-002',
                'llm_model': 'llama3.1',
                'vector_store_ready': True
            }
        """
        info = {
            'setup_complete': all([self.vectorstore, self.retriever, self.rag_chain]),
            'documents_loaded': len(self.documents) if self.documents else 0,
            'chunks_created': len(self.doc_splits) if self.doc_splits else 0,
            'config_file': self.config_manager.config_path,
            'embedding_model': self.config['embeddings']['model'],
            'embedding_provider': self.config['embeddings']['provider'],
            'llm_model': self.config['llm']['model'],
            'llm_provider': self.config['llm']['provider'],
            'chunk_size': self.config['processing']['chunk_size'],
            'chunk_overlap': self.config['processing']['chunk_overlap'],
            'retrieval_k': self.config['retrieval']['k'],
            'vector_store_ready': self.vectorstore is not None,
            'categorization_enabled': self.config['categories']['enabled']
        }
        return info
    
    def export_processed_data(self, output_path: str = "processed_data.json"):
        """
        Export processed documents and their metadata for inspection.
        
        This is useful for:
        1. Debugging data processing issues
        2. Understanding how data was chunked
        3. Validating metadata extraction
        4. Creating test datasets
        
        Args:
            output_path (str): Path to save the exported data
            
        Raises:
            ValueError: If no data has been processed yet
        """
        if not self.doc_splits:
            raise ValueError("No processed data to export. Run setup() first.")
        
        # Prepare export data
        export_data = {
            'summary': {
                'total_documents': len(self.documents),
                'total_chunks': len(self.doc_splits),
                'processing_config': self.config['processing'],
                'export_timestamp': str(datetime.now())
            },
            'original_documents': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in self.documents
            ],
            'processed_chunks': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'chunk_id': i
                }
                for i, doc in enumerate(self.doc_splits)
            ]
        }
        
        # Save to file
        import datetime
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported processed data to {output_path}")


def main():
    """
    Main function providing comprehensive CLI interface.
    
    Command-line options:
    --config: Path to configuration YAML file
    --data: Path to JSON data file (overrides config)
    --setup-only: Only setup the system, don't run interactive mode
    --query: Run a single query and exit
    --export: Export processed data to file
    --info: Show system information
    
    Examples:
        # Basic usage with default config
        python rag_app.py
        
        # Use custom config and data files
        python rag_app.py --config my_config.yaml --data ocean_data.json
        
        # Run single query
        python rag_app.py --query "What temperature sensors are available?"
        
        # Setup only (useful for testing configuration)
        python rag_app.py --setup-only
        
        # Export processed data for inspection
        python rag_app.py --export processed_ocean_data.json
    """
    parser = argparse.ArgumentParser(
        description="Generic RAG Application for JSON Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode with default config
  %(prog)s --config ocean_config.yaml        # Use custom configuration
  %(prog)s --data sensors.json               # Use custom data file
  %(prog)s --query "Show temperature data"   # Single query mode
  %(prog)s --setup-only                      # Setup only, no interaction
  %(prog)s --info                           # Show system information
  %(prog)s --export processed.json          # Export processed data

For SENG 499 ONC Project - Ocean Data Assistant Foundation
        """
    )
    
    # Define command-line arguments
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--data", 
        help="Path to JSON data file (overrides config setting)"
    )
    parser.add_argument(
        "--setup-only", 
        action="store_true",
        help="Only setup the system, don't run interactive mode"
    )
    parser.add_argument(
        "--query", 
        help="Run a single query and exit (for testing/automation)"
    )
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Show system information and configuration"
    )
    parser.add_argument(
        "--export", 
        help="Export processed data to specified file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize the RAG application
        print("Initializing Generic RAG Application...")
        app = GenericRAGApplication(args.config)
        
        # Setup the application
        print("Setting up data processing pipeline...")
        app.setup(args.data)
        
        # Handle different execution modes
        if args.info:
            # Show system information
            info = app.get_system_info()
            print("\nSYSTEM INFORMATION")
            print("="*50)
            for key, value in info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        elif args.export:
            # Export processed data
            app.export_processed_data(args.export)
            print(f"Processed data exported to {args.export}")
        
        elif args.query:
            # Single query mode
            print(f"\nQuestion: {args.query}")
            answer = app.query(args.query)
            print(f"Answer: {answer}")
        
        elif not args.setup_only:
            # Interactive mode (default)
            app.interactive_mode()
        else:
            # Setup only mode
            print("Setup complete! System ready for queries.")
            print(f"{len(app.documents)} documents loaded, {len(app.doc_splits)} chunks created")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: File not found - {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected Error: {e}")
        print("Please check the logs for more details.")
        return 1
    
    return 0

# Script execution guard
if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    This ensures the main() function only runs when the script is executed
    directly, not when imported as a module.
    """
    exit(main())