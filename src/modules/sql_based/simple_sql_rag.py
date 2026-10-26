import psycopg2

from src.architectures.sql_rag import SQLRAG
from src.components.llm import SyncOpenAILikeLLM
from src.components.sql_query_executor import UnsafePostgresSQLExecutor
from src.components.sql_schema_extractors import PostgresSQLSchemaExtractor
from src.settings import settings

conn = psycopg2.connect(**settings.postgres.connect_kwargs())

try:
    executor = UnsafePostgresSQLExecutor(conn)
    extractor = PostgresSQLSchemaExtractor(conn)
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
