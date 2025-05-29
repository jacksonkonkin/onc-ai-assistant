# ONC AI Assistant

A specialized Retrieval-Augmented Generation (RAG) pipeline designed for Ocean Networks Canada (ONC) data and documentation. This AI assistant helps researchers, students, and the public understand oceanographic data, instruments, and marine observations.

## Features

- **Persistent Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- **Multi-LLM Support**: Compatible with Groq, OpenAI, and Ollama
- **ONC Document Integration**: Automatically processes ONC documentation and links
- **Arctic Observatory Focus**: Specialized knowledge of Cambridge Bay and other ONC observatories
- **Instrument Expertise**: Understands CTD, hydrophones, ADCP, cameras, and other marine instruments
- **Cost-Effective**: Persistent embeddings reduce API costs on subsequent runs

## Quick Start

### 1. Environment Setup

Create a `.env` file with your API keys:
```bash
# Copy the template and fill in your keys
cp .env.template .env

# Required: At least one LLM API key
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional: Logging and directories
LOG_LEVEL=INFO
ONC_DOCUMENT_DIR=onc_documents
ONC_VECTOR_STORE_DIR=onc_vectorstore
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Assistant

#### First Run (Downloads and processes documents)
```bash
python onc_rag_pipeline.py --download
```

#### Interactive Mode
```bash
python onc_rag_pipeline.py
```

#### Single Query Mode
```bash
python onc_rag_pipeline.py --query "What is a CTD instrument?"
```

## Document Sources

The pipeline automatically processes:

- **ONC Wiki Pages**: Instrument documentation, data products, observatory information
- **Links File**: URLs from `onc_documents/links.txt` including:
  - Ocean Networks Canada main site
  - Oceans 3.0 data portal and API
  - Cambridge Bay observatory documentation
  - Oceanographic terminology glossaries
- **Local Documents**: PDFs, HTML, markdown, and text files

## Configuration

Edit `onc_config.yaml` to customize:

```yaml
# LLM Provider (groq, openai, ollama)
llm:
  provider: groq
  groq:
    model: llama3-70b-8192
    temperature: 0.1

# Vector Store Settings
vector_store:
  provider: chroma          # chroma (persistent) or sklearn (temporary)
  persist_directory: onc_vectorstore
  force_rebuild: false      # Set to true to rebuild embeddings
  collection_name: onc_documents

# Document Processing
processing:
  chunk_size: 500
  chunk_overlap: 50
  batch_size: 20

# Retrieval Settings
retrieval:
  k: 8                      # Number of documents to retrieve
  similarity_threshold: 0.75
```

## Architecture

1. **Document Ingestion**: Downloads ONC documentation and processes local files
2. **Text Processing**: Splits documents into semantic chunks with metadata
3. **Vector Embeddings**: Creates OpenAI embeddings with ChromaDB persistence  
4. **LLM Integration**: Routes queries through Groq, OpenAI, or Ollama
5. **Retrieval**: Semantic search finds relevant ONC documents
6. **Response Generation**: Context-aware answers using ONC-specific prompts

## Supported Document Types

- **PDF**: Research papers, technical documentation
- **HTML**: Wiki pages, web documentation  
- **Markdown**: Technical notes, documentation
- **Text**: Data descriptions, instrument specifications

## Example Queries

- "What instruments are deployed at Cambridge Bay?"
- "How does a CTD measure salinity?"
- "What are the ice conditions in the Arctic observatories?"
- "Explain the Oceans 3.0 API for data access"
- "What marine mammals are detected by hydrophones?"
