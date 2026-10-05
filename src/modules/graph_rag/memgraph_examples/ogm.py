from gqlalchemy import Memgraph
from gqlalchemy.models import Node, Relationship, Field
from datetime import date

memgraph = Memgraph(host="127.0.0.1", port=7600, username="memgraph", password="memgraph-password")
memgraph.execute("MATCH (n) DETACH DELETE n")

class User(Node):
    name: str = Field(unique=True, db=memgraph)
    age: int | None = None

class Friend(Relationship):
    since: date | None = None


u1 = User(name="Alice", age=23).save(memgraph)
print(u1)
u2 = User(name="Bob").save(memgraph)
print(u2)

r = Friend(_start_node_id=u1._id, _end_node_id=u2._id, since=date(year=2026, month=1, day=1)).save(memgraph)
print(r)

alice = User(name="Alice").load(memgraph)
alice.age = 25
alice.save(memgraph)
print(alice)