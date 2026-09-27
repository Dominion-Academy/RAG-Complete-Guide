from neo4j import GraphDatabase, NotificationMinimumSeverity

from src.architectures.graph_rag import GraphRAG
from src.components.graph_query_executor import Neo4jCypherExecutor
from src.components.graph_schema_extractors import Neo4jSchemaExtractor
from src.components.llm import SyncOpenAILikeLLM
from src.settings import settings


driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "neo4j-password"),
    notifications_min_severity=NotificationMinimumSeverity.OFF,
)

try:
    executor = Neo4jCypherExecutor(driver)
    extractor = Neo4jSchemaExtractor(driver)
    llm = SyncOpenAILikeLLM(
        base_url=settings.llm.BASE_URL,
        api_key=settings.llm.API_KEY,
        model=settings.llm.MODEL,
        common_parameters={"temperature": 0},
    )
    graph_rag = GraphRAG(executor, extractor, llm)
    answer = graph_rag.generate("Какие есть люди?")
    print(answer.content)
finally:
    driver.close()
