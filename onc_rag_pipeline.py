#!/usr/bin/env python3
"""
ONC-Specific RAG Pipeline for Ocean Networks Canada
==================================================

This pipeline integrates Groq and Llama APIs with OpenAI embeddings to create
a specialized RAG system for Ocean Networks Canada documents and data.

Features:
- Pre-download and process ONC documents
- Support for multiple LLM providers (Groq, Ollama, OpenAI)
- OpenAI embeddings for semantic search
- ONC-specific prompt templates and oceanographic knowledge
- Document categorization by type (instruments, data, research papers)

Author: SENG 499 AI & NLP Team
Purpose: ONC Ocean Data Assistant with enhanced LLM support
"""

import os
import json
import yaml
import requests
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import argparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM providers
from groq import Groq

# Document processing
import PyPDF2
from bs4 import BeautifulSoup
import tiktoken

# Configure logging with environment variable override
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_environment():
    """
    Validate that required environment variables are set.
    
    Returns:
        bool: True if all required variables are set, False otherwise
    """
    required_vars = []
    warnings = []
    
    # Check for at least one LLM API key
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not groq_key and not openai_key:
        required_vars.append("At least one of GROQ_API_KEY or OPENAI_API_KEY must be set")
    
    # Check for OpenAI key if using OpenAI embeddings (default)
    if not openai_key:
        warnings.append("OPENAI_API_KEY not set - OpenAI embeddings will not work")
    
    # Log validation results
    if required_vars:
        logger.error("Missing required environment variables:")
        for var in required_vars:
            logger.error(f"  - {var}")
        return False
    
    if warnings:
        logger.warning("Environment warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    logger.info("Environment validation passed")
    return True


def load_env_with_fallback():
    """
    Load environment variables with fallback to .env file.
    """
    # Try to load from .env file in current directory
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment variables from .env file")
    else:
        logger.info("No .env file found, using system environment variables")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Please check your .env file or environment variables.")
        logger.info("You can copy .env.template to .env and fill in your API keys")
        return False
    
    return True


class ONCDocumentDownloader:
    """
    Downloads and preprocesses ONC documents from various sources.
    
    Supports:
    - ONC API endpoints
    - Direct PDF downloads
    - Web scraping of ONC documentation
    - Local document processing
    """
    
    def __init__(self, download_dir: str = None):
        """
        Initialize the document downloader.
        
        Args:
            download_dir (str): Directory to store downloaded documents
        """
        # Use environment variable or default
        if download_dir is None:
            download_dir = os.getenv('ONC_DOCUMENT_DIR', 'onc_documents')
        
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # ONC document sources (simplified for Sprint 1)
        self.document_sources = {
            "instrument_docs": [
                "https://wiki.oceannetworks.ca/display/O2A/CTD",
                "https://wiki.oceannetworks.ca/display/O2A/Hydrophone"
            ],
            "observatory_info": [
                "https://wiki.oceannetworks.ca/display/O2A/Cambridge+Bay"
            ]
        }
    
    def process_links_file(self, links_file: str = "links.txt") -> List[str]:
        """
        Process URLs from a links file and download their content.
        
        Args:
            links_file (str): Path to file containing URLs
            
        Returns:
            List[str]: List of downloaded file paths
        """
        links_path = self.download_dir / links_file
        if not links_path.exists():
            logger.warning(f"Links file {links_path} not found")
            return []
        
        downloaded_files = []
        
        with open(links_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract URLs from the content
        import re
        urls = re.findall(r'https?://[^\s]+', content)
        
        logger.info(f"Found {len(urls)} URLs in {links_file}")
        
        for url in urls:
            try:
                file_path = self._download_document(url, "links_file")
                if file_path:
                    downloaded_files.append(file_path)
                    logger.info(f"✓ Downloaded from links: {url}")
            except Exception as e:
                logger.error(f"✗ Failed to download {url}: {e}")
                continue
        
        return downloaded_files
    
    def download_all_documents(self) -> Dict[str, List[str]]:
        """
        Download all configured ONC documents.
        
        Returns:
            Dict[str, List[str]]: Mapping of document types to downloaded file paths
        """
        downloaded_files = {}
        
        for doc_type, urls in self.document_sources.items():
            logger.info(f"Downloading {doc_type} documents...")
            downloaded_files[doc_type] = []
            
            for url in urls:
                try:
                    file_path = self._download_document(url, doc_type)
                    if file_path:
                        downloaded_files[doc_type].append(file_path)
                        logger.info(f"✓ Downloaded: {file_path}")
                except Exception as e:
                    logger.error(f"✗ Failed to download {url}: {e}")
                    continue
        
        # Process links.txt file if it exists
        links_files = self.process_links_file()
        if links_files:
            downloaded_files["links_file"] = links_files
        
        return downloaded_files
    
    def _download_document(self, url: str, doc_type: str) -> Optional[str]:
        """
        Download a single document from URL.
        
        Args:
            url (str): Document URL
            doc_type (str): Document category
            
        Returns:
            Optional[str]: Path to downloaded file or None if failed
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type and name
            if url.endswith('.pdf'):
                filename = f"{doc_type}_{url.split('/')[-1]}"
                file_path = self.download_dir / filename
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                # HTML content - sanitize filename
                url_part = url.split('/')[-1].replace('/', '_').replace('?', '_').replace('#', '_')
                if not url_part or url_part == '_':
                    url_part = f"page_{hash(url) % 10000}"
                filename = f"{doc_type}_{url_part}.html"
                file_path = self.download_dir / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def process_local_documents(self, local_dir: str) -> List[str]:
        """
        Process documents from a local directory.
        
        Args:
            local_dir (str): Path to directory containing ONC documents
            
        Returns:
            List[str]: List of processed document paths
        """
        local_path = Path(local_dir)
        if not local_path.exists():
            logger.warning(f"Local directory {local_dir} does not exist")
            return []
        
        processed_files = []
        for file_path in local_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.html', '.md']:
                processed_files.append(str(file_path))
        
        # Also look for links.txt in the local directory
        links_file = local_path / 'links.txt'
        if links_file.exists():
            logger.info(f"Found links.txt in {local_dir}")
            # This will be processed by the downloader's process_links_file method
        
        logger.info(f"Found {len(processed_files)} local documents")
        return processed_files


class ONCDocumentProcessor:
    """
    Processes ONC documents into LangChain Document objects with ONC-specific metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document processor.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple documents into LangChain Document objects.
        
        Args:
            file_paths (List[str]): List of document file paths
            
        Returns:
            List[Document]: Processed documents with metadata
        """
        documents = []
        
        for file_path in file_paths:
            try:
                docs = self._process_single_document(file_path)
                documents.extend(docs)
                logger.info(f"✓ Processed {file_path}: {len(docs)} document chunks")
            except Exception as e:
                logger.error(f"✗ Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Total documents processed: {len(documents)}")
        return documents
    
    def _process_single_document(self, file_path: str) -> List[Document]:
        """
        Process a single document file.
        
        Args:
            file_path (str): Path to document file
            
        Returns:
            List[Document]: List of document chunks
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif file_path.suffix.lower() in ['.html', '.htm']:
            return self._process_html(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self._process_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
    
    def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process PDF document."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            return self._create_documents(text, file_path, "pdf")
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def _process_html(self, file_path: Path) -> List[Document]:
        """Process HTML document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                
            return self._create_documents(text, file_path, "html")
        except Exception as e:
            logger.error(f"Error processing HTML {file_path}: {e}")
            return []
    
    def _process_text(self, file_path: Path) -> List[Document]:
        """Process text document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return self._create_documents(text, file_path, "text")
        except Exception as e:
            logger.error(f"Error processing text {file_path}: {e}")
            return []
    
    def _create_documents(self, text: str, file_path: Path, doc_type: str) -> List[Document]:
        """
        Create Document objects with ONC-specific metadata.
        
        Args:
            text (str): Document text content
            file_path (Path): Original file path
            doc_type (str): Document type
            
        Returns:
            List[Document]: List of document chunks
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Determine ONC document category
        onc_category = self._categorize_onc_document(file_path.name, text)
        
        # Extract ONC-specific metadata
        metadata = {
            "source": str(file_path),
            "doc_type": doc_type,
            "onc_category": onc_category,
            "filename": file_path.name,
            "file_size": len(text),
            "processed_at": datetime.now().isoformat(),
            "token_count": len(self.encoding.encode(text))
        }
        
        # Add instrument-specific metadata if applicable
        if onc_category == "instrument":
            metadata.update(self._extract_instrument_metadata(text))
        
        return [Document(page_content=text, metadata=metadata)]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize document text."""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Join with single newlines
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text
    
    def _categorize_onc_document(self, filename: str, text: str) -> str:
        """
        Categorize ONC document based on filename and content.
        
        Args:
            filename (str): Document filename
            text (str): Document content
            
        Returns:
            str: Document category
        """
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        # Instrument documentation
        instruments = ['ctd', 'hydrophone', 'adcp', 'camera', 'seismometer', 'accelerometer']
        if any(inst in filename_lower or inst in text_lower for inst in instruments):
            return "instrument"
        
        # Observatory information
        observatories = ['cambridge bay', 'folger passage', 'strait of georgia', 'barkley canyon']
        if any(obs in filename_lower or obs in text_lower for obs in observatories):
            return "observatory"
        
        # Data products
        data_keywords = ['data product', 'data quality', 'calibration', 'processing']
        if any(kw in filename_lower or kw in text_lower for kw in data_keywords):
            return "data_product"
        
        # Research papers/publications
        research_keywords = ['research', 'publication', 'paper', 'study', 'analysis']
        if any(kw in filename_lower or kw in text_lower for kw in research_keywords):
            return "research"
        
        return "general"
    
    def _extract_instrument_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract basic instrument metadata from text (simplified for Sprint 1).
        """
        metadata = {}
        text_lower = text.lower()
        
        # Basic instrument detection
        if any(term in text_lower for term in ['temperature', 'salinity', 'pressure']):
            metadata['oceanographic_instrument'] = True
        if any(term in text_lower for term in ['acoustic', 'sound', 'hydrophone']):
            metadata['acoustic_instrument'] = True
        
        return metadata


class GroqLLMWrapper:
    """
    Wrapper for Groq API to work with LangChain-style interfaces.
    """
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192", temperature: float = 0.1):
        """
        Initialize Groq LLM wrapper.
        
        Args:
            api_key (str): Groq API key
            model (str): Model name
            temperature (float): Generation temperature
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
    
    def invoke(self, prompt_dict: Dict[str, str]) -> str:
        """
        Invoke the Groq model with a prompt.
        
        Args:
            prompt_dict (Dict[str, str]): Dictionary containing prompt components
            
        Returns:
            str: Generated response
        """
        try:
            # Format the prompt
            if isinstance(prompt_dict, dict) and 'text' in prompt_dict:
                messages = [{"role": "user", "content": prompt_dict['text']}]
            else:
                # Assume it's a formatted prompt string
                messages = [{"role": "user", "content": str(prompt_dict)}]
            
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error with Groq API: {e}")
            return f"Error generating response: {str(e)}"


class ONCRAGPipeline:
    """
    Main ONC RAG Pipeline integrating multiple LLM providers with OpenAI embeddings.
    """
    
    def __init__(self, config_path: str = "onc_config.yaml"):
        """
        Initialize the ONC RAG pipeline.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.document_downloader = ONCDocumentDownloader(
            self.config.get('document_dir', 'onc_documents')
        )
        self.document_processor = ONCDocumentProcessor(self.config)
        
        # Initialize empty attributes
        self.documents = None
        self.doc_splits = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with ONC-specific defaults and environment variable overrides."""
        default_config = {
            "document_dir": os.getenv('ONC_DOCUMENT_DIR', 'onc_documents'),
            "llm": {
                "provider": "groq",
                "groq": {
                    "model": os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile'),
                    "temperature": 0.1,
                    "api_key_env": "GROQ_API_KEY"
                }
            },
            "embeddings": {
                "provider": "openai",
                "model": os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'),
                "api_key_env": "OPENAI_API_KEY"
            },
            "processing": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "batch_size": 20
            },
            "retrieval": {
                "k": 8,
                "similarity_threshold": 0.75
            },
            "vector_store": {
                "provider": "chroma",
                "persist_directory": os.getenv('ONC_VECTOR_STORE_DIR', 'onc_vectorstore'),
                "force_rebuild": False,
                "collection_name": "onc_documents"
            },
            "onc_specific": {
                "cambridge_bay_focus": True,
                "instrument_priority": ["CTD", "hydrophone", "ADCP"],
                "data_types": ["temperature", "salinity", "pressure", "acoustic"]
            }
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                    # Deep merge configurations
                    self._deep_update(default_config, user_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.info("Using default configuration")
        else:
            # Create default config file
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default config file: {self.config_path}")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update of nested dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup(self, download_new: bool = True, local_docs_dir: Optional[str] = None):
        """
        Setup the complete ONC RAG pipeline.
        
        Args:
            download_new (bool): Whether to download new documents
            local_docs_dir (Optional[str]): Path to local documents directory
        """
        logger.info("Setting up ONC RAG Pipeline...")
        
        # Step 1: Get documents
        document_files = []
        
        if download_new:
            logger.info("Downloading ONC documents...")
            downloaded = self.document_downloader.download_all_documents()
            for doc_type, files in downloaded.items():
                document_files.extend(files)
        
        # Auto-detect local documents if none specified but directory exists
        if not local_docs_dir:
            default_doc_dir = self.config.get('document_dir', 'onc_documents')
            if Path(default_doc_dir).exists():
                local_docs_dir = default_doc_dir
                logger.info(f"Auto-detected local documents directory: {local_docs_dir}")
        
        if local_docs_dir:
            logger.info(f"Processing local documents from {local_docs_dir}")
            local_files = self.document_downloader.process_local_documents(local_docs_dir)
            document_files.extend(local_files)
        
        if not document_files:
            logger.warning("No documents found. Will use direct LLM mode with ONC prompt template.")
            self.documents = []
            self.doc_splits = []
            self.vectorstore = None
            self.retriever = None
            
            # Setup LLM for direct mode
            logger.info("Setting up LLM for direct mode...")
            self._setup_llm()
            
            # Create direct response chain
            logger.info("Creating direct response chain...")
            self._setup_direct_chain()
            
            logger.info("ONC RAG Pipeline setup complete (direct LLM mode)!")
            return
        
        # Step 2: Process documents
        logger.info("Processing documents...")
        self.documents = self.document_processor.process_documents(document_files)
        
        # Step 3: Split documents
        logger.info("Splitting documents into chunks...")
        self._setup_text_splitter()
        self.doc_splits = self.text_splitter.split_documents(self.documents)
        logger.info(f"Created {len(self.doc_splits)} document chunks")
        
        # Step 4: Setup embeddings and vector store
        logger.info("Setting up persistent Chroma vector store...")
        self._setup_vectorstore()
        
        # Step 5: Setup LLM
        logger.info("Setting up LLM...")
        self._setup_llm()
        
        # Step 6: Create RAG chain
        logger.info("Creating RAG chain...")
        self._setup_rag_chain()
        
        logger.info("ONC RAG Pipeline setup complete!")
    
    def _setup_text_splitter(self):
        """Setup text splitter for ONC documents."""
        processing_config = self.config['processing']
        
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=processing_config['chunk_size'],
            chunk_overlap=processing_config['chunk_overlap'],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _setup_vectorstore(self):
        """Setup vector store with persistence support."""
        embeddings_config = self.config['embeddings']
        vector_config = self.config['vector_store']
        
        api_key = os.getenv(embeddings_config['api_key_env'])
        if not api_key:
            raise ValueError(f"Environment variable {embeddings_config['api_key_env']} not set")
        
        embedding_function = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=embeddings_config['model']
        )
        
        persist_dir = Path(vector_config['persist_directory'])
        
        # Check if persistent store exists and is not forced to rebuild
        if (persist_dir.exists() and 
            not vector_config['force_rebuild'] and
            len(list(persist_dir.iterdir())) > 0):
            
            logger.info(f"Loading existing vector store from {persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embedding_function,
                collection_name=vector_config['collection_name']
            )
            logger.info(f"Loaded {self.vectorstore._collection.count()} documents from persistent store")
        
        else:
            # Create new vector store
            logger.info("Creating new persistent vector store...")
            persist_dir.mkdir(exist_ok=True)
            
            batch_size = self.config['processing']['batch_size']
            self.vectorstore = self._create_chroma_vectorstore(
                self.doc_splits, embedding_function, batch_size, str(persist_dir),
                vector_config['collection_name']
            )
        
        # Setup retriever with dynamic k based on available documents
        doc_count = self.vectorstore._collection.count() if hasattr(self.vectorstore, '_collection') else len(self.doc_splits)
        max_k = min(self.config['retrieval']['k'], doc_count)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": max_k}
        )
    
    
    def _create_chroma_vectorstore(self, doc_splits: List[Document], 
                                 embedding_function, batch_size: int,
                                 persist_directory: str, collection_name: str):
        """Create Chroma vector store with persistence."""
        logger.info(f"Creating Chroma vector store with {len(doc_splits)} documents")
        
        # Create initial vectorstore with first batch
        first_batch = doc_splits[:batch_size]
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Process remaining batches
        for i in range(batch_size, len(doc_splits), batch_size):
            batch = doc_splits[i:i + batch_size]
            
            try:
                vectorstore.add_documents(batch)
                logger.info(f"✓ Added batch {i//batch_size + 1}/{(len(doc_splits)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"✗ Error in batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Chroma vector store created with {vectorstore._collection.count()} documents")
        return vectorstore
    
    def add_documents_to_vectorstore(self, new_documents: List[Document]):
        """Add new documents to existing vector store."""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return
        
        if not new_documents:
            logger.info("No new documents to add")
            return
        
        logger.info(f"Adding {len(new_documents)} new documents to vector store")
        
        try:
            if hasattr(self.vectorstore, 'add_documents'):
                self.vectorstore.add_documents(new_documents)
                logger.info(f"✓ Successfully added {len(new_documents)} documents")
            else:
                logger.warning("Vector store does not support incremental updates")
        except Exception as e:
            logger.error(f"✗ Error adding documents: {e}")
    
    def rebuild_vectorstore(self):
        """Force rebuild of the vector store."""
        logger.info("Forcing vector store rebuild...")
        self.config['vector_store']['force_rebuild'] = True
        self._setup_vectorstore()
        self.config['vector_store']['force_rebuild'] = False  # Reset for next time
    
    def _setup_llm(self):
        """Setup Groq LLM."""
        llm_config = self.config['llm']
        
        api_key = os.getenv(llm_config['groq']['api_key_env'])
        if not api_key:
            raise ValueError(f"Groq API key not found in environment variable {llm_config['groq']['api_key_env']}")
        
        self.llm = GroqLLMWrapper(
            api_key=api_key,
            model=llm_config['groq']['model'],
            temperature=llm_config['groq']['temperature']
        )
        logger.info(f"Using Groq LLM: {llm_config['groq']['model']}")
    
    def _setup_rag_chain(self):
        """Setup the RAG processing chain with ONC-specific prompts."""
        # ONC-specific prompt template
        onc_prompt = PromptTemplate(
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

CONTEXT FROM ONC DOCUMENTS:
{documents}

USER QUESTION: {question}

EXPERT ONC ANALYSIS:""",
            input_variables=["question", "documents"]
        )
        
        # Create Groq processing chain
        def groq_chain(inputs):
            formatted_prompt = onc_prompt.format(**inputs)
            response = self.llm.invoke({"text": formatted_prompt})
            return response
        
        self.rag_chain = groq_chain
    
    def _setup_direct_chain(self):
        """Setup direct LLM chain when no documents are available."""
        # ONC-specific prompt template for direct responses
        onc_direct_prompt = PromptTemplate(
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

NOTE: No specific ONC documents are currently loaded, so responses are based on general oceanographic knowledge.

USER QUESTION: {question}

EXPERT ONC ANALYSIS:""",
            input_variables=["question"]
        )
        
        # Create direct Groq processing chain
        def groq_direct_chain(inputs):
            formatted_prompt = onc_direct_prompt.format(**inputs)
            response = self.llm.invoke({"text": formatted_prompt})
            return response
        
        self.rag_chain = groq_direct_chain
    
    def query(self, question: str) -> str:
        """
        Process an ONC-related query.
        
        Args:
            question (str): User question about ONC data or instruments
            
        Returns:
            str: Generated response based on ONC documents or direct LLM
        """
        if not self.rag_chain:
            raise ValueError("Pipeline not setup. Call setup() first.")
        
        try:
            logger.info(f"Processing ONC query: {question[:100]}...")
            
            # Check if we're in RAG mode (have documents) or direct mode
            if self.retriever and self.documents:
                # RAG mode - retrieve and use documents
                documents = self.retriever.invoke(question)
                logger.info(f"Retrieved {len(documents)} relevant ONC documents")
                
                # Format documents with ONC-specific metadata
                doc_texts = []
                for i, doc in enumerate(documents):
                    source = doc.metadata.get('filename', f'Document_{i+1}')
                    category = doc.metadata.get('onc_category', 'general')
                    doc_type = doc.metadata.get('doc_type', 'unknown')
                    
                    header = f"[{source}] (Type: {category}, Format: {doc_type})"
                    doc_texts.append(f"{header}\n{doc.page_content}")
                
                combined_docs = "\n\n" + "="*60 + "\n\n".join(doc_texts)
                
                # Generate response with documents using Groq
                answer = self.rag_chain({
                    "question": question,
                    "documents": combined_docs
                })
            else:
                # Direct mode - use Groq LLM without documents
                logger.info("Using direct LLM mode (no documents loaded)")
                
                answer = self.rag_chain({
                    "question": question
                })
            
            logger.info("Query processed successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error processing your ONC question: {str(e)}"
    
    def interactive_mode(self):
        """Run interactive ONC Q&A mode."""
        print("\n" + "="*70)
        print("ONC OCEAN DATA ASSISTANT")
        print("Specialized AI for Ocean Networks Canada")
        print("="*70)
        
        # Show mode information
        if self.documents and len(self.documents) > 0:
            print(f"Mode: RAG (Document-based)")
            print(f"Loaded: {len(self.documents)} ONC documents")
            print(f"Searchable chunks: {len(self.doc_splits)}")
            print(f"Embeddings: {self.config['embeddings']['provider']}")
        else:
            print(f"Mode: Direct LLM (No documents loaded)")
            print(f"Using general oceanographic knowledge")
        
        print(f"LLM: {self.config['llm']['provider']}")
        print("-"*70)
        print("Ask about:")
        print("   • Cambridge Bay observations and instruments")
        print("   • CTD, hydrophone, and ADCP data")
        print("   • Ocean temperature, salinity, and ice conditions")
        print("   • Marine mammals and acoustic monitoring")
        print("   • Data products and quality information")
        print("-"*70)
        print("Type 'quit' to exit")
        print("="*70 + "\n")
        
        question_count = 0
        
        while True:
            try:
                question = input("Ask about ONC data: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print(f"\nThank you for using the ONC Ocean Data Assistant!")
                    if question_count > 0:
                        print(f"You asked {question_count} question{'s' if question_count != 1 else ''}.")
                    break
                
                question_count += 1
                print("Analyzing ONC documents...", end='', flush=True)
                
                answer = self.query(question)
                
                print("\r" + " "*40 + "\r", end='')
                print(f"**ONC Assistant:** {answer}\n")
                print("─" * 50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {str(e)}")
                print("Please try again.\n")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="ONC RAG Pipeline - Ocean Networks Canada AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (can be set in .env file):
  GROQ_API_KEY          Groq API key for LLM access
  OPENAI_API_KEY        OpenAI API key for embeddings and LLM
  ONC_API_TOKEN         ONC API token (optional)
  LOG_LEVEL             Logging level (DEBUG, INFO, WARNING, ERROR)
  ONC_DOCUMENT_DIR      Directory for storing documents
  GROQ_MODEL            Groq model name override
  OPENAI_EMBEDDING_MODEL OpenAI embedding model override

Examples:
  python onc_rag_pipeline.py --download
  python onc_rag_pipeline.py --local-docs /path/to/docs
  python onc_rag_pipeline.py --query "What is a CTD?"
        """
    )
    
    parser.add_argument("--config", default="onc_config.yaml", help="Configuration file path")
    parser.add_argument("--download", action="store_true", help="Download new ONC documents")
    parser.add_argument("--local-docs", help="Path to local ONC documents directory")
    parser.add_argument("--query", help="Single query mode")
    parser.add_argument("--setup-only", action="store_true", help="Setup only, no interaction")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--env-check", action="store_true", help="Check environment variables and exit")
    parser.add_argument("--rebuild-vectorstore", action="store_true", help="Force rebuild vector store")
    
    args = parser.parse_args()
    
    # Load environment variables
    print("Loading environment variables...")
    if not load_env_with_fallback():
        print("Environment setup failed. Please check your API keys in .env file.")
        return 1
    
    if args.env_check:
        print("Environment variables validated successfully!")
        return 0
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print("Initializing ONC RAG Pipeline...")
        pipeline = ONCRAGPipeline(args.config)
        
        # Handle special arguments
        if args.rebuild_vectorstore:
            print("Forcing vector store rebuild...")
            pipeline.config['vector_store']['force_rebuild'] = True
        
        print("Setting up pipeline...")
        pipeline.setup(download_new=args.download, local_docs_dir=args.local_docs)
        
        if args.query:
            print(f"\nQuestion: {args.query}")
            answer = pipeline.query(args.query)
            print(f"Answer: {answer}")
        elif not args.setup_only:
            pipeline.interactive_mode()
        else:
            print("Setup complete! ONC RAG Pipeline ready.")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())