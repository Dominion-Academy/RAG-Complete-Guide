import os
from gqlalchemy import Memgraph, create, match
from gqlalchemy.models import Node, Field
from datetime import date

from gqlalchemy.query_builders.declarative_base import Operator, Order


memgraph = Memgraph(host="127.0.0.1", port=7600, username="memgraph", password="memgraph-password")
memgraph.execute("MATCH (n) DETACH DELETE n")


# Создание узлов
create(connection=memgraph).node(labels="User", name="Alice", age=23).execute()
create(connection=memgraph).node(labels="User", name="Bob").execute()

# Создание связи
(
    match(connection=memgraph).node(labels="User", name="Alice", variable="a")
    .match().node(labels="User", name="Bob", variable="b")
    .create()
    .node(variable="a")
    .to(relationship_type="FRIEND", since=date(year=2026, month=1, day=1))
    .node(variable="b")
    .execute()
)

# Обновление узла
(
    match(connection=memgraph)
    .node(variable="n")
    .where(item="n.name", operator=Operator.EQUAL, literal="Alice")
    .set_(item="n.age", operator=Operator.ASSIGNMENT, literal=25)
    .execute()
)

# Схема для создания Python-объектов
class User(Node):
    name: str = Field(unique=True, db=memgraph)
    age: int | None = None

# Создание узла через User-объект
nick = User(name="Nick", age=19)
create(connection=memgraph).node(node=nick).execute()

# Получение User-объектов
users = list(
    match(connection=memgraph)
    .node(labels="User", variable="u")
    .return_("u")
    .order_by(properties=("u.name", Order.DESC))
    .limit(2)
    .execute()
)
print(users)

# Удаление узла
(
    match(connection=memgraph)
    .node(labels="User", name="Nick", variable="u")
    .delete(variable_expressions="u")
    .execute()
)


# Удаление отношение
(
    match(connection=memgraph)
    .node(labels="User", name="Alice").to(relationship_type="FRIEND", variable="r").node(labels="User")
    .delete(variable_expressions="r")
    .execute()
)

# Получение оставшихся User-объектов
users = list(
    match(connection=memgraph)
    .node(labels="User", variable="u")
    .return_("u")
    .execute()
)
print(users)