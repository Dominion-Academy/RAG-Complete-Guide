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

| Module              | Topic                 | Examples                                                                                                                                                                                                                                                                                                                                                                                          |
|---------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Introduction**     | First RAG with ngrams | [First RAG with ngrams](src/modules/first_rag/readme.md)                                                                                                                                                                                                                                                                                                                                          |
| **2. Vector Based RAG** | BERT Embeddings       | [Transformers](src/modules/vector_search/bert_embeddings/transformers_usage.py), [Sentence Transformers](src/modules/vector_search/bert_embeddings/sentence_transformers_usage.py), [FastEmbed](src/modules/vector_search/bert_embeddings/fastembed_usage.py), [Ollama](src/modules/vector_search/bert_embeddings/ollama_usage.py)                                                                |
|                     | FAISS                 | [Simple Example](src/modules/vector_search/vector_databases/faiss_example.py), [HNSW Example](src/modules/vector_search/vector_databases/faiss_hnsw_example.py)                                                                                                                                                                                                                                   |
|                     | SQLite + sqlite-vec   | [Simple Example](src/modules/vector_search/vector_databases/sqlite_vec_example.py)                                                                                                                                                                                                                                                                                                                |
|                     | PostgreSQL + pgvector | [Simple Example](src/modules/vector_search/vector_databases/pgvector_example.py), [HNSW Example](src/modules/vector_search/vector_databases/pgvector_hnsw_example.py)                                                                                                                                                                                                                             |
|                     | Qdrant                | [Simple Example](src/modules/vector_search/vector_databases/qdrant_example.py), [HNSW Example](src/modules/vector_search/vector_databases/qdrant_hnsw_example.py)                                                                                                                                                                                                                                 |
|                     | Chroma                | [Simple Example](src/modules/vector_search/vector_databases/chroma_example.py), [Advanced Example](src/modules/vector_search/vector_databases/chroma_advanced_example.py)                                                                                                                                                                                                                         |
|                     | Milvus                | [Simple Example](src/modules/vector_search/vector_databases/milvus_example.py), [HNSW Example](src/modules/vector_search/vector_databases/milvus_hnsw_example.py)                                                                                                                                                                                                                                 |
|                     | Canonical RAG         | [Code](src/modules/vector_search/canonical_rag)                                                                                                                                                                                                                                                                                                                                                   |
|                     | Text Splitters        | [By paragraph](src/modules/vector_search/text_spliiters_demo/split_by_paragraph_demo.py), [With overlap](src/modules/vector_search/text_spliiters_demo/split_by_chars_with_overlap_demo.py), [Recursive](src/modules/vector_search/text_spliiters_demo/recursive_split_demo.py)                                                                                                                   |
|                     | Pre-Retrieval         | [Multi Query Rewriting](src/modules/vector_search/pre_retrieval/multi_query_rewriting.py), [Step Back Prompting](src/modules/vector_search/pre_retrieval/step_back_prompting.py), [HyDE](src/modules/vector_search/pre_retrieval/hyde.py)                                                                                                                                                         |
|                     | Retrieval             | [BM25](src/modules/vector_search/retrieval/bm25_in_memory.py), [BM25 in Qdrant](src/modules/vector_search/retrieval/bm25_in_qdrant.py), [Hybrid Search](src/modules/vector_search/retrieval/hybrid_search.py), [SPLADE in Qdrant](src/modules/vector_search/retrieval/splade_in_qdrant.py), [Hybrid RAG in Qdrant Full Pipeline](src/modules/vector_search/retrieval/hybrid_rag_full_pipeline.py) |
|                     | Post-Retrieval        | [Fusion RAG in Qdrant Full Pipeline](src/modules/vector_search/post_retrieval/fusion_rag_full_pipeline.py)                                                                                                                                                                                                                                                                                             |


## Scripts

### Linters

```bash
uv run ruff format src && uv run ruff check --fix src
```