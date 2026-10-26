from abc import ABC, abstractmethod

from psycopg2.extensions import connection
from psycopg2.extras import RealDictCursor


class BaseSQLExecutor(ABC):
    @abstractmethod
    def execute(self, query: str) -> list[dict]:
        raise NotImplementedError


class UnsafePostgresSQLExecutor(BaseSQLExecutor):
    def __init__(self, conn: connection):
        self._conn = conn

    def execute(self, query: str) -> list[dict]:
        with self._conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            records = [dict(row) for row in cursor.fetchall()] if cursor.description else []
        self._conn.commit()
        return records


class PostgresSQLExecutor(BaseSQLExecutor):
    def __init__(self, conn: connection):
        self._conn = conn

    def execute(self, query: str) -> list[dict]:
        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                records = [dict(row) for row in cursor.fetchall()] if cursor.description else []
            self._conn.rollback()
            return records
        except Exception:
            self._conn.rollback()
            raise
