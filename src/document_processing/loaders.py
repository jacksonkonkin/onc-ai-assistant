"""
Document loading utilities
Handles file system operations and document discovery
Team: Data Engineering
"""

import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles loading documents from various sources."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.html', '.htm', '.md'}
    
    def __init__(self, base_dir: str = 'onc_documents'):
        """Initialize document loader."""
        self.base_dir = Path(base_dir)
    
    def load_local_documents(self, doc_dir: str = None) -> List[str]:
        """
        Load documents from local directory.
        
        Args:
            doc_dir (str): Directory to search. Uses base_dir if None.
            
        Returns:
            List[str]: List of document file paths
        """
        if doc_dir is None:
            doc_dir = self.base_dir
        else:
            doc_dir = Path(doc_dir)
        
        if not doc_dir.exists():
            logger.warning(f"Document directory {doc_dir} does not exist")
            return []
        
        files = []
        for file_path in doc_dir.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS):
                files.append(str(file_path))
        
        logger.info(f"Found {len(files)} documents in {doc_dir}")
        return files
    
    def validate_document_path(self, file_path: str) -> bool:
        """
        Validate if a document path is supported.
        
        Args:
            file_path (str): Path to validate
            
        Returns:
            bool: True if supported
        """
        path = Path(file_path)
        return (path.exists() and 
                path.is_file() and 
                path.suffix.lower() in self.SUPPORTED_EXTENSIONS)