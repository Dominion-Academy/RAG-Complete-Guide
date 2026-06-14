# Naive RAG

## Architecture

```mermaid
flowchart LR
    Query(Query) --> Retriever[Retriever]
    Retriever --> KnowledgeStorage[Knowledge Storage]
    Retriever --> ExternalKnowledge(External Knowledge)
    ExternalKnowledge --> LLM[LLM]
    Query -.-> LLM
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
python -m src.modules.first_rag.app -n 3 -k 2 --query "Почему спасатели не могут приблизиться к семи молодым Ки-Лир, впавшим в кататонию в заброшенной шахте?"
```
> Ожидаемый ответ:
> Потому что стены шахты были натёрты базальтовой пылью до зеркального блеска, и любой, кто приблизится, рискует вторичным отражением — то есть тоже увидит «Пустоту внутри» и впадёт в кататонию.


## Retrieval evaluation

```bash
python -m src.modules.first_rag.eval -n 3 -k 2
```

```plain
===== FINAL RESULTS =====
N queries: 25
Hit Rate@2: 1.00
```