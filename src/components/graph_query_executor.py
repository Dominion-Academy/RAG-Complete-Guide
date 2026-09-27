from abc import ABC, abstractmethod

from neo4j import Driver


class BaseCypherExecutor(ABC):
    @abstractmethod
    def execute(self, query: str) -> list[dict]:
        raise NotImplementedError


class Neo4jCypherExecutor(BaseCypherExecutor):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def execute(self, query: str) -> list[dict]:
        records, _, _ = self._driver.execute_query(
            query_=query,
            database_=self._database,
        )
        return [record.data() for record in records]
