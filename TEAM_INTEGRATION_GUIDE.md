# ONC AI Assistant - Sprint 2 Team Integration Guide

## Overview

The ONC RAG pipeline has been modularized to enable cross-functional collaboration across teams. Each module has clear interfaces and responsibilities aligned with the Sprint 2 features.

## Architecture

```
src/
├── config/           # Configuration management
├── document_processing/  # Data Engineering team
├── vector_database/     # Data + LLM teams  
├── query_routing/       # Backend + Data teams
├── database_search/     # Backend + Data + LLM teams
├── rag_engine/         # LLM team
└── api/               # Backend team (main integration point)
```

## Team Responsibilities & Integration Points

### 1. Frontend Team
**Feature: Connect frontend to backend**

**Integration Points:**
- `src/api/pipeline.py` - Main ONCPipeline class
- Entry point: `onc_rag_pipeline_modular.py`

**Key Methods:**
```python
from src.api.pipeline import ONCPipeline

# Initialize
pipeline = ONCPipeline("onc_config.yaml")
pipeline.setup(doc_dir="onc_documents")

# Query processing
response = pipeline.query("What is the temperature at Cambridge Bay?")

# Status checking
status = pipeline.get_pipeline_status()
```

### 2. Backend Team  
**Features: Query Routing, Ocean 3 database search**

**Integration Points:**
- `src/query_routing/router.py` - QueryRouter class
- `src/database_search/ocean3_client.py` - Ocean3Client class
- `src/api/pipeline.py` - Main orchestration

**Key Areas for Development:**
```python
# Query routing customization
from src.query_routing import QueryRouter

router = QueryRouter(routing_config)
routing_decision = router.route_query(query, context)

# Ocean 3 database integration (currently placeholder)
from src.database_search import Ocean3Client

client = Ocean3Client(db_config)
results = client.search_observations(query_params)
```

### 3. Data Engineering Team
**Features: Document processing, Vector database operations**

**Integration Points:**
- `src/document_processing/` - All document processing logic
- `src/vector_database/vector_store.py` - Vector operations

**Key Classes:**
```python
# Document processing
from src.document_processing import DocumentProcessor, DocumentLoader

processor = DocumentProcessor()
documents = processor.process_documents(file_paths)

# Vector database management  
from src.vector_database import VectorStoreManager

vector_manager = VectorStoreManager(vector_config, processing_config, embedding_manager)
vector_manager.setup_vectorstore(documents)
```

### 4. LLM/AI Team
**Features: RAG enhancements, Vector database search**

**Integration Points:**
- `src/rag_engine/` - Core RAG functionality
- `src/vector_database/embeddings.py` - Embedding management

**Key Classes:**
```python
# RAG engine
from src.rag_engine import RAGEngine, LLMWrapper

llm_wrapper = LLMWrapper(llm_config)
rag_engine = RAGEngine(llm_wrapper)

# Process different query types
response = rag_engine.process_rag_query(question, documents)
response = rag_engine.process_direct_query(question)
response = rag_engine.process_hybrid_query(question, vector_docs, db_results)
```

## Configuration Management

All teams should use the centralized configuration:

```python
from src.config.settings import ConfigManager

config = ConfigManager("onc_config.yaml")
llm_config = config.get_llm_config()
vector_config = config.get_vector_store_config()
```

## Testing the Modular Implementation

### Basic Functionality Test
```bash
# Test with existing data
python onc_rag_pipeline_modular.py --docs onc_documents

# Test single query
python onc_rag_pipeline_modular.py --query "What instruments are at Cambridge Bay?"

# Force rebuild vector store
python onc_rag_pipeline_modular.py --force-rebuild
```

### Team-Specific Testing

**Frontend Team:**
```python
# Test pipeline integration
from src.api.pipeline import ONCPipeline

pipeline = ONCPipeline()
pipeline.setup()
status = pipeline.get_pipeline_status()
response = pipeline.query("test query")
```

**Backend Team:**
```python
# Test query routing
from src.query_routing import QueryRouter

router = QueryRouter()
decision = router.route_query("latest temperature data")
print(decision['type'])  # Should route to database_search

# Test Ocean 3 client (currently returns mock data)
from src.database_search import Ocean3Client

client = Ocean3Client()
results = client.search_observations({})
```

**Data Team:**
```python
# Test document processing
from src.document_processing import DocumentProcessor

processor = DocumentProcessor()
docs = processor.process_documents(['path/to/doc.pdf'])

# Test vector operations
from src.vector_database import VectorStoreManager, EmbeddingManager

embedding_manager = EmbeddingManager(embeddings_config)
vector_manager = VectorStoreManager(vector_config, processing_config, embedding_manager)
```

**LLM Team:**
```python
# Test RAG engine components
from src.rag_engine import RAGEngine, LLMWrapper

llm_wrapper = LLMWrapper(llm_config)
rag_engine = RAGEngine(llm_wrapper)

response = rag_engine.process_direct_query("What is oceanography?")
```

## Migration from Original Pipeline

The original `onc_rag_pipeline.py` is preserved. The new modular system:

1. **Maintains same functionality** - All original features work
2. **Same configuration** - Uses existing `onc_config.yaml`
3. **Same entry point pattern** - Similar command-line interface
4. **Backwards compatible** - Existing scripts should work with minimal changes

## Next Steps for Sprint 2

### Immediate (Week 1)
1. **Frontend Team**: Integrate with `ONCPipeline` class
2. **Backend Team**: Implement Ocean 3 database connection in `Ocean3Client`
3. **Data Team**: Enhance document processing and vector operations
4. **LLM Team**: Implement RAG enhancements and fine-tune prompts

### Integration (Week 2-3)
1. Connect frontend to modular backend
2. Implement real Ocean 3 database queries
3. Add advanced query routing logic
4. Enhance vector search capabilities

### Testing & Deployment (Week 3-4)
1. End-to-end integration testing
2. Performance optimization
3. Error handling and monitoring
4. Documentation and deployment

## Support

Each module includes comprehensive logging and error handling. For debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The modular design allows teams to work independently while maintaining integration through well-defined interfaces.