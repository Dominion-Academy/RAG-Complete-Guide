from pprint import pprint

from neo4j import GraphDatabase


driver = GraphDatabase.driver(
    "bolt://localhost:7600",
)

driver.verify_connectivity()

with driver.session() as session:
    result = session.run("""
SHOW SCHEMA INFO
    """)
    record = result.single()
    pprint(record)
