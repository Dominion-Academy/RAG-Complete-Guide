import psycopg2

from src.architectures.sql_rag import SQLRAG
from src.components.embeddings import FastEmbedEmbeddingsModel
from src.components.llm import SyncOpenAILikeLLM
from src.components.rerankers import FastEmbedReranker
from src.components.sql_query_executor import UnsafePostgresSQLExecutor
from src.components.sql_schema_extractors import SEMANTIC_EMBEDDINGS_MODEL, PostgresSQLSemanticContextExtractor
from src.settings import settings

conn = psycopg2.connect(**settings.postgres.connect_kwargs())

try:
    executor = UnsafePostgresSQLExecutor(conn)
    extractor = PostgresSQLSemanticContextExtractor(
        conn,
        embeddings_model=FastEmbedEmbeddingsModel(SEMANTIC_EMBEDDINGS_MODEL),
        reranker=FastEmbedReranker("jinaai/jina-reranker-v2-base-multilingual"),
    )
    llm = SyncOpenAILikeLLM(
        base_url=settings.llm.BASE_URL,
        api_key=settings.llm.API_KEY,
        model=settings.llm.MODEL,
        common_parameters={"temperature": 0},
    )
    sql_rag = SQLRAG(executor, extractor, llm)
    answer = sql_rag.generate("Какая выручка у постоянных клиентов из Германии за 2025 год?")
    print("\nОтвет: ", answer.content)
finally:
    conn.close()
