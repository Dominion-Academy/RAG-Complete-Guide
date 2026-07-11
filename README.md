# RAG Complete Guide

## Structure

- Vector Search RAG
- Data Processing
- Graph RAG
- Table RAG
- Multimodal RAG
- Agentic RAG
- Web Search RAG
- etc.

## Getting started

```bash
uv venv --python 3.14
uv sync
```

## Detail Program

| Модуль | Тема | Ссылки                                                                                                                                                                                                                                                                                                                            |
|--------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction | First RAG with ngrams | [First RAG with ngrams](src/modules/first_rag/readme.md)                                                                                                                                                                                                                                                                          |
| Vector Based RAG | BERT Embeddings | [Transformers](src/modules/vector_search/bert_embeddings/transformers_usage.py), [Sentence Transformers](src/modules/vector_search/bert_embeddings/sentence_transformers_usage.py), [FastEmbed](src/modules/vector_search/bert_embeddings/fastembed_usage.py), [Ollama](src/modules/vector_search/bert_embeddings/ollama_usage.py) |
| | FAISS | [Simple Example](src/modules/vector_search/vector_databases/faiss_example.py), [HNSW Example](src/modules/vector_search/vector_databases/faiss_hnsw_example.py)                                                                                                                                                                   |
| | SQLite + sqlite-vec | [Simple Example](src/modules/vector_search/vector_databases/sqlite_vec_example.py)                                                                                                                                                                                                                                                |
| | PostgreSQL + pgvector | [Simple Example](src/modules/vector_search/vector_databases/pgvector_example.py), [HNSW Example](src/modules/vector_search/vector_databases/pgvector_hnsw_example.py)                                                                                                                                                             |
| | Qdrant | [Simple Example](src/modules/vector_search/vector_databases/qdrant_example.py), [HNSW Example](src/modules/vector_search/vector_databases/qdrant_hnsw_example.py)                                                                                                                                                                 |
| | Chroma | [Simple Example](src/modules/vector_search/vector_databases/chroma_example.py), [Advanced Example](src/modules/vector_search/vector_databases/chroma_advanced_example.py)                                                                                                                                                         |
| | Milvus | [Simple Example](src/modules/vector_search/vector_databases/milvus_example.py), [HNSW Example](src/modules/vector_search/vector_databases/milvus_hnsw_example.py)                                                                                                                                                                 |
| | Canonical RAG | [Code](src/modules/vector_search/canonical_rag)                                                                                                                                                                                                                                                                                   |


## Scripts

### Linters

```bash
uv run ruff format src && uv run ruff check --fix src
```