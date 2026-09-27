from abc import ABC, abstractmethod

from neo4j import Driver


class BaseSchemaExtractor(ABC):
    @abstractmethod
    def extract_schemas(self) -> tuple[dict, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_schema_text(self) -> str:
        raise NotImplementedError


class Neo4jSchemaExtractor(BaseSchemaExtractor):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def extract_schemas(self) -> tuple[dict, dict]:
        node_records, _, _ = self._driver.execute_query("CALL db.schema.nodeTypeProperties()", database_=self._database)
        rel_records, _, _ = self._driver.execute_query("CALL db.schema.relTypeProperties()", database_=self._database)
        return self._build_nodes(node_records), self._build_rels(rel_records)

    @staticmethod
    def _build_nodes(records) -> dict:
        nodes: dict = {}
        for record in records:
            node_type = record["nodeType"]
            nodes.setdefault(node_type, {})
            property_name = record["propertyName"]
            property_types = record["propertyTypes"]
            if property_name and property_types:
                nodes[node_type][property_name] = property_types[0]
        return nodes

    @staticmethod
    def _build_rels(records) -> dict:
        rels: dict = {}
        for record in records:
            rel_type = record["relType"]
            rels.setdefault(rel_type, {})
            property_name = record["propertyName"]
            property_types = record["propertyTypes"]
            if property_name and property_types:
                rels[rel_type][property_name] = property_types[0]
        return rels

    def get_schema_text(self) -> str:
        nodes, rels = self.extract_schemas()

        node_desc = "\n".join(f"{node}{props}" for node, props in nodes.items())
        rel_desc = "\n".join(f"{rel}{props}" for rel, props in rels.items())

        return f"Node labels and properties: \n{node_desc}\n\nRelationship types and properties: \n{rel_desc}\n"
