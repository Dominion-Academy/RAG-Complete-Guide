from neo4j import GraphDatabase


driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "neo4j-password"))

records, _, _ = driver.execute_query("MATCH (p:Person) RETURN count(p) AS count")
print(records[0]["count"])
# records, _, _ = driver.execute_query("MATCH (p:Person) RETURN p")
# print(records[0]["p"]["name"])

driver.close()
