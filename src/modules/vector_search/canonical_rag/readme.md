# Canonical RAG

## Architecture

```mermaid
flowchart LR
    Document(Document) --> Chunker[split_by_words]
    Chunker --> Chunks(Chunks)
    Chunks --> Embedder[EmbeddingsModel]
    Embedder --> Embeddings(Embeddings)
    Embeddings --> KnowledgeStorage[QdrantCollectionKnowledgeStorage]
    
    classDef doc fill:#f59e0b,stroke:#f59e0b,stroke-width:2px,color:#fff
    classDef chunker fill:#E38EC3,stroke:#E38EC3,stroke-width:2px,color:#fff
    classDef chunks fill:#F3C6E2,stroke:#F3C6E2,stroke-width:2px,color:#000
    classDef embedder fill:#60A5FA,stroke:#60A5FA,stroke-width:2px,color:#fff
    classDef embeddings fill:#34D399,stroke:#34D399,stroke-width:2px,color:#fff
    classDef storage fill:#8B5CF6,stroke:#8B5CF6,stroke-width:2px,color:#fff

    class Document doc
    class Chunker chunker
    class Chunks chunks
    class Embedder embedder
    class Embeddings embeddings
    class KnowledgeStorage storage
```


```mermaid
flowchart LR
    Query(Query) --> Retriever[SimpleRetriever]
    Retriever -.-> KnowledgeStorage[QdrantCollectionKnowledgeStorage]
    Retriever --> ExternalKnowledge(External Knowledge)
    ExternalKnowledge --> LLM[LLM]
    Query --> LLM
    LLM --> Answer(Answer)
    
    classDef q fill:#f59e0b,stroke:#f59e0b,stroke-width:2px,color:#fff
    classDef a fill:#10b981,stroke:#10b981,stroke-width:2px,color:#fff
    classDef ks fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff
    classDef r fill:#E38EC3,stroke:#E38EC3,stroke-width:2px,color:#fff
    classDef ks fill:#D7CFF9,stroke:#D7CFF9,stroke-width:2px,color:#000
    classDef ek fill:#F3C6E2,stroke:#F3C6E2,stroke-width:2px,color:#000
    classDef llm fill:#F9C6C6,stroke:#F9C6C6,stroke-width:2px,color:#000

    class Query q
    class Retriever r
    class KnowledgeStorage ks
    class ExternalKnowledge ek
    class LLM llm
    class Answer a
```

## Running

```bash
python -m src.modules.vector_search.canonical_rag.run
```
> Ожидаемый ответ:
> The Berry Export Summary 2028 is a dedicated export plan for the Australian strawberry, raspberry, and blackberry industries. It maps the sectors’ current position, where they want to be, high-opportunity markets, and next steps. The purpose of this plan is to grow their global presence over the next 10 years.
