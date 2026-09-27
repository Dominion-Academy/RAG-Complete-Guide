from neo4j import GraphDatabase, NotificationMinimumSeverity

from src.components.graph_schema_extractors import Neo4jSchemaExtractor


driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "neo4j-password"),
    notifications_min_severity=NotificationMinimumSeverity.OFF,
)

try:
    extractor = Neo4jSchemaExtractor(driver)
    nodes_schema, rels_schema = extractor.extract_schemas()
    print(extractor.get_schema_text())
finally:
    driver.close()
