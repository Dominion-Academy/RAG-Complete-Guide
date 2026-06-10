# Naive RAG

## Architecture

```mermaid
flowchart LR
    Query[Query] --> Retriever[Retriever]
    Retriever --> KnowledgeStorage[Knowledge Storage]
    Retriever --> ExternalKnowledge[External Knowledge]
    ExternalKnowledge --> LLM[LLM]
    Query -.-> LLM
    LLM --> Answer[Answer]
    
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

### Eng
```bash
python -m src.modules.01_intro.naive_rag.app --lang eng --query "Why can't rescuers approach the seven young Qi-Lir who fell into catatonia in the abandoned mine?"
```

> Expected answer:
> Because the walls were polished with basalt dust to a mirror shine, and anyone who approaches risks secondary reflection — that is, also seeing "the Void Within" and falling catatonic.

### Ru
```bash
python -m src.modules.01_intro.naive_rag.app --lang ru --query "Почему спасатели не могут приблизиться к семи молодым Ки-Лир, впавшим в кататонию в заброшенной шахте?"
```
> Ожидаемый ответ:
> Потому что стены шахты были натёрты базальтовой пылью до зеркального блеска, и любой, кто приблизится, рискует вторичным отражением — то есть тоже увидит «Пустоту внутри» и впадёт в кататонию.
