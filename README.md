# RAG Complete Guide

<img width="1210" height="333" alt="image" src="https://github.com/user-attachments/assets/0dfdc50e-f6c2-47ac-b0e5-9b22e9bd144a" />


## Main topics

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
uv venv --python 3.12
uv sync
```

## Detail Program

| Module                  | Topic                 | Examples                                                                                                                                                                                                                                                                                                                                                                                         |
|-------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Introduction**     | First RAG with ngrams | [First RAG with ngrams](src/modules/first_rag/readme.md)                                                                                                                                                                                                                                                                                                                                         |
| **2. Vector Based RAG** | BERT Embeddings       | [Transformers](src/modules/vector_search/bert_embeddings/transformers_usage.py), [Sentence Transformers](src/modules/vector_search/bert_embeddings/sentence_transformers_usage.py), [FastEmbed](src/modules/vector_search/bert_embeddings/fastembed_usage.py), [Ollama](src/modules/vector_search/bert_embeddings/ollama_usage.py)                                                               |
|                         | FAISS                 | [Simple Example](src/modules/vector_search/vector_databases/faiss_example.py), [HNSW Example](src/modules/vector_search/vector_databases/faiss_hnsw_example.py)                                                                                                                                                                                                                                  |
|                         | SQLite + sqlite-vec   | [Simple Example](src/modules/vector_search/vector_databases/sqlite_vec_example.py)                                                                                                                                                                                                                                                                                                               |
|                         | PostgreSQL + pgvector | [Simple Example](src/modules/vector_search/vector_databases/pgvector_example.py), [HNSW Example](src/modules/vector_search/vector_databases/pgvector_hnsw_example.py)                                                                                                                                                                                                                            |
|                         | Qdrant                | [Simple Example](src/modules/vector_search/vector_databases/qdrant_example.py), [HNSW Example](src/modules/vector_search/vector_databases/qdrant_hnsw_example.py)                                                                                                                                                                                                                                |
|                         | Chroma                | [Simple Example](src/modules/vector_search/vector_databases/chroma_example.py), [Advanced Example](src/modules/vector_search/vector_databases/chroma_advanced_example.py)                                                                                                                                                                                                                        |
|                         | Milvus                | [Simple Example](src/modules/vector_search/vector_databases/milvus_example.py), [HNSW Example](src/modules/vector_search/vector_databases/milvus_hnsw_example.py)                                                                                                                                                                                                                                |
|                         | Canonical RAG         | [Code](src/modules/vector_search/canonical_rag)                                                                                                                                                                                                                                                                                                                                                  |
|                         | Text Splitters        | [By paragraph](src/modules/vector_search/text_spliiters_demo/split_by_paragraph_demo.py), [With overlap](src/modules/vector_search/text_spliiters_demo/split_by_chars_with_overlap_demo.py), [Recursive](src/modules/vector_search/text_spliiters_demo/recursive_split_demo.py)                                                                                                                  |
|                         | Pre-Retrieval         | [Multi Query Rewriting](src/modules/vector_search/pre_retrieval/multi_query_rewriting.py), [Step Back Prompting](src/modules/vector_search/pre_retrieval/step_back_prompting.py), [HyDE](src/modules/vector_search/pre_retrieval/hyde.py)                                                                                                                                                        |
|                         | Retrieval             | [BM25](src/modules/vector_search/retrieval/bm25_in_memory.py), [BM25 in Qdrant](src/modules/vector_search/retrieval/bm25_in_qdrant.py), [Hybrid Search](src/modules/vector_search/retrieval/hybrid_search.py), [SPLADE in Qdrant](src/modules/vector_search/retrieval/splade_in_qdrant.py), [Hybrid RAG in Qdrant Full Pipeline](src/modules/vector_search/retrieval/hybrid_rag_full_pipeline.py) |
|                         | Post-Retrieval        | [Fusion RAG in Qdrant Full Pipeline](src/modules/vector_search/post_retrieval/fusion_rag_full_pipeline.py)                                                                                                                                                                                                                                                                                       |
|                         | Prompting             | [Prompt](src/components/prompts.py)                                                                                                                                                                                                                                                                                                                                                              |
|                         | Metadata filtering    | [Metadata Extraction](src/modules/vector_search/extra/metadata_extraction.py), [Metadata Filtering](src/modules/vector_search/extra/metadata_filtering.py)                                                                                                                                                                                                                                       |
| **3. Evaluation**       | Simple RAG Pipeline   | [Simple RAG for Evaluation](src/modules/evaluation/rag_pipelines/simple.py)                                                                                                                                                                                                                                                                                                                      |
|                         | RAGAS                 | [Dataset](src/modules/evaluation/ragas_evals/create_dataset.py), [Metrics](src/modules/evaluation/ragas_evals/estimate_context_precision.py), [Experiments](src/modules/evaluation/ragas_evals/first_experiment.py)                                                                                                                                                                              |
|                         | DeepEval              | [First Test](src/modules/evaluation/deepeval_evals/test_first.py), [RAG Evaluation](src/modules/evaluation/deepeval_evals/rag_evaluation.py), [Golden Dataset](src/modules/evaluation/deepeval_evals/evaluate_golden_dataset.py), [Synthetic Dataset](src/modules/evaluation/deepeval_evals/generate_golden_dataset.py)                                                                          |
|                         | MLflow                | [Golden Dataset](src/modules/evaluation/mlflow_experiments/create_dataset.py), [RAG Evaluation](src/modules/evaluation/mlflow_experiments/run_experiment.py), [Custom Metrics Evaluation](src/modules/evaluation/mlflow_experiments/run_with_custom_metrics.py)                                                                                                                                  |
| **4. GraphRAG**         | Neo4j                 | [First Query](src/modules/graph_rag/neo4j_examples/first_query.py), [Schema Extraction](src/modules/graph_rag/neo4j_examples/schema_extraction.py), [GraphRAG Pipeline](src/modules/graph_rag/neo4j_examples/graphrag_pipeline.py), [Graph Knowledge Extration](src/modules/graph_rag/neo4j_examples/extract_graph_from_text.py)                                                                 |
|        | Memgraph              | [First Query](src/modules/graph_rag/memgraph_examples/first_query.py), [OGM](src/modules/graph_rag/memgraph_examples/ogm.py), [Query Building](src/modules/graph_rag/memgraph_examples/query_building.py)                                                                                 |


## Scripts

### Linters

```bash
uv run ruff format src && uv run ruff check --fix src
```
