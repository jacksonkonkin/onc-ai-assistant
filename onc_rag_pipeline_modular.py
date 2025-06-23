#!/usr/bin/env python3
"""
Modular ONC RAG Pipeline
========================

Entry point for the modularized ONC RAG Pipeline system.
Maintains compatibility with the original pipeline while providing
a clean, modular architecture for cross-team collaboration.

Teams and Responsibilities:
- Frontend team: Connect to API endpoints (src/api/)
- Backend team: Query routing, Ocean 3 DB integration, API layer
- Data Engineering team: Document processing (src/document_processing/)
- LLM/AI team: RAG engine, vector database, embeddings (src/rag_engine/, src/vector_database/)

Author: SENG 499 AI & NLP Team
"""

import os
import argparse
import logging
from typing import Optional

# Ensure src is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.pipeline import ONCPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_environment():
    """Validate that required environment variables are set."""
    groq_key = os.getenv('GROQ_API_KEY')
    mistral_key = os.getenv('MISTRAL_API_KEY')
    
    if not groq_key:
        logger.error("GROQ_API_KEY is required")
        return False
    
    if not mistral_key:
        logger.error("MISTRAL_API_KEY is required for embeddings")
        return False
    
    logger.info("Environment validation passed")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Modular ONC RAG Pipeline")
    parser.add_argument("--docs", default="onc_documents", help="Documents directory")
    parser.add_argument("--config", default="onc_config.yaml", help="Configuration file")
    parser.add_argument("--query", help="Single query mode")
    parser.add_argument("--force-rebuild", action="store_true", 
                       help="Force rebuild of vector store")
    
    args = parser.parse_args()
    
    # Check environment
    if not validate_environment():
        print("Please set GROQ_API_KEY and MISTRAL_API_KEY environment variables")
        return 1
    
    try:
        print("Initializing Modular ONC RAG Pipeline...")
        
        # Initialize pipeline with configuration
        pipeline = ONCPipeline(args.config)
        
        # Setup pipeline
        success = pipeline.setup(args.docs, args.force_rebuild)
        
        if not success:
            print("Pipeline setup failed")
            return 1
        
        # Show pipeline status
        status = pipeline.get_pipeline_status()
        print(f"\nPipeline Status:")
        print(f"  Documents: {status['document_count']} loaded")
        print(f"  Vector Store: {'Ready' if status['vector_store_ready'] else 'Not available'}")
        print(f"  Database: {'Connected' if status['database_connected'] else 'Not connected'}")
        print(f"  LLM: {status['llm_info'].get('provider')} - {status['llm_info'].get('model')}")
        
        if args.query:
            # Single query mode
            print(f"\nProcessing query: {args.query}")
            answer = pipeline.query(args.query)
            print(f"\nAnswer: {answer}")
        else:
            # Interactive mode
            pipeline.interactive_mode()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())