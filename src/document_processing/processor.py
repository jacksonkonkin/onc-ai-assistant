"""
Document processing core functionality
Handles conversion of documents to LangChain Document objects
Team: Data Engineering
"""

import logging
from typing import List
from pathlib import Path

import PyPDF2
from bs4 import BeautifulSoup
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents into LangChain Document objects."""
    
    def __init__(self):
        """Initialize document processor."""
        pass
    
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
        """Create Document objects with metadata."""
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Enhanced metadata for cross-team collaboration
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "doc_type": doc_type,
            "file_size": file_path.stat().st_size,
            "last_modified": file_path.stat().st_mtime
        }
        
        return [Document(page_content=text, metadata=metadata)]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize document text."""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Join with single newlines
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text