from abc import ABC, abstractmethod

from psycopg2.extensions import connection


class BaseSQLSchemaExtractor(ABC):
    @abstractmethod
    def extract_schema(self) -> dict[str, dict[str, str]]:
        """Возвращает {table_name: {column_name: data_type}}"""
        raise NotImplementedError

    @abstractmethod
    def get_schema_text(self) -> str:
        raise NotImplementedError


class PostgresSQLSchemaExtractor(BaseSQLSchemaExtractor):
    COLUMNS_QUERY = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
    """
    FOREIGN_KEYS_QUERY = """
        SELECT
            tc.table_name, kcu.column_name,
            ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s
        ORDER BY tc.table_name, kcu.column_name
    """

    def __init__(self, conn: connection, schema: str = "public"):
        self._conn = conn
        self._schema = schema

    def _fetch(self, query: str) -> list[tuple]:
        try:
            with self._conn.cursor() as cursor:
                cursor.execute(query, (self._schema,))
                return cursor.fetchall()
        finally:
            self._conn.rollback()

    def extract_schema(self) -> dict[str, dict[str, str]]:
        tables: dict[str, dict[str, str]] = {}
        for table_name, column_name, data_type in self._fetch(self.COLUMNS_QUERY):
            tables.setdefault(table_name, {})[column_name] = data_type
        return tables

    def extract_foreign_keys(self) -> list[tuple[str, str, str, str]]:
        return self._fetch(self.FOREIGN_KEYS_QUERY)

    def get_schema_text(self) -> str:
        tables = self.extract_schema()
        tables_desc = "\n".join(
            f"{table}({', '.join(f'{column} {data_type}' for column, data_type in columns.items())})"
            for table, columns in tables.items()
        )
        schema_text = f"Tables and columns:\n{tables_desc}\n"
        foreign_keys = self.extract_foreign_keys()
        if foreign_keys:
            fk_desc = "\n".join(f"{t}.{c} -> {ft}.{fc}" for t, c, ft, fc in foreign_keys)
            schema_text += f"\nForeign keys:\n{fk_desc}\n"
        return schema_text
