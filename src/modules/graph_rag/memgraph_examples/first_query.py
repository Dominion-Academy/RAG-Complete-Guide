from gqlalchemy import Memgraph

memgraph = Memgraph(host="127.0.0.1", port=7600, username="memgraph", password="memgraph-password")

memgraph.execute("MATCH (n) DETACH DELETE n")

query = """
CREATE (n:FirstNode)
SET n.message = 'Hello, World!'
RETURN n.message AS result
"""
results = memgraph.execute_and_fetch(query)
print(list(results)[0]['result'])