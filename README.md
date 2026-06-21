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

### Introduction

- [First RAG with ngrams](src/modules/first_rag/readme.md)

### Vector Based RAG

#### BERT Embeddings usage
- [Transformers Example](src/modules/vector_search/bert_embeddings/transformers_usage.py)
- [Sentence Transformers Example](src/modules/vector_search/bert_embeddings/sentence_transformers_usage.py)
- [FastEmbed Example](src/modules/vector_search/bert_embeddings/fastembed_usage.py)
- [Ollama Example](src/modules/vector_search/bert_embeddings/ollama_usage.py)



## Scripts

### Linters

```bash
uv run ruff format src && uv run ruff check --fix src
```