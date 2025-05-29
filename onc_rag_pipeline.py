#!/usr/bin/env python3
"""
Simplified ONC RAG Pipeline
==========================

Core RAG functionality for Ocean Networks Canada documents:
- Local document processing (PDF, HTML, text files)
- OpenAI embeddings for semantic search  
- Chroma vector database storage
- Groq LLM for question answering

Author: SENG 499 AI & NLP Team
"""

import os
import logging
from typing import List, Optional
from pathlib import Path
import argparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# LLM providers
from groq import Groq

# Document processing
import PyPDF2
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_environment():
    """Validate that required environment variables are set."""
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not groq_key:
        logger.error("GROQ_API_KEY is required")
        return False
    
    if not openai_key:
        logger.error("OPENAI_API_KEY is required for embeddings")
        return False
    
    logger.info("Environment validation passed")
    return True


def load_local_documents(doc_dir: str = 'onc_documents') -> List[str]:
    """Load documents from local directory."""
    doc_path = Path(doc_dir)
    if not doc_path.exists():
        logger.warning(f"Document directory {doc_dir} does not exist")
        return []
    
    files = []
    for file_path in doc_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.html', '.md']:
            files.append(str(file_path))
    
    logger.info(f"Found {len(files)} documents in {doc_dir}")
    return files


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
        """Create Document objects with basic metadata."""
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Basic metadata
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "doc_type": doc_type
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
    


class GroqLLMWrapper:
    """Simple Groq API wrapper."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def invoke(self, prompt: str) -> str:
        """Generate response from prompt."""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error: {str(e)}"


class SimpleRAGPipeline:
    """Simple RAG pipeline for ONC documents."""
    
    def __init__(self, doc_dir: str = 'onc_documents'):
        self.doc_dir = doc_dir
        self.document_processor = DocumentProcessor()
        
        # Simple config
        self.config = {
            'processing': {'chunk_size': 500, 'chunk_overlap': 50, 'batch_size': 20},
            'embeddings': {'provider': 'openai', 'model': 'text-embedding-ada-002', 'api_key_env': 'OPENAI_API_KEY'},
            'vector_store': {'provider': 'chroma', 'persist_directory': 'onc_vectorstore', 'force_rebuild': False, 'collection_name': 'onc_documents'},
            'retrieval': {'k': 8, 'similarity_threshold': 0.75},
            'llm': {'provider': 'groq', 'groq': {'model': 'llama-3.3-70b-versatile', 'temperature': 0.1, 'api_key_env': 'GROQ_API_KEY'}}
        }
        
        # Initialize empty attributes
        self.documents = None
        self.doc_splits = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
    
    
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
        
        # Auto-detect local documents if none specified but directory exists
        if not local_docs_dir:
            default_doc_dir = self.doc_dir
            if Path(default_doc_dir).exists():
                local_docs_dir = default_doc_dir
                logger.info(f"Auto-detected local documents directory: {local_docs_dir}")
        
        if local_docs_dir:
            logger.info(f"Processing local documents from {local_docs_dir}")
            local_files = load_local_documents(local_docs_dir)
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
            model=llm_config['groq']['model']
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
            response = self.llm.invoke(formatted_prompt)
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
            response = self.llm.invoke(formatted_prompt)
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
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple ONC RAG Pipeline")
    parser.add_argument("--docs", default="onc_documents", help="Documents directory")
    parser.add_argument("--query", help="Single query mode")
    
    args = parser.parse_args()
    
    # Check environment
    if not validate_environment():
        print("Please set GROQ_API_KEY and OPENAI_API_KEY environment variables")
        return 1
    
    try:
        print("Initializing RAG Pipeline...")
        pipeline = SimpleRAGPipeline(args.docs)
        pipeline.setup()
        
        if args.query:
            answer = pipeline.query(args.query)
            print(f"Answer: {answer}")
        else:
            pipeline.interactive_mode()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())