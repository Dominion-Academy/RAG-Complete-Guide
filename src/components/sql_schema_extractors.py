import re
from abc import ABC, abstractmethod
from collections import deque
from itertools import combinations
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from psycopg2.extensions import connection

from src.components.embeddings import BaseEmbeddingsModel
from src.components.knowledge_storage import Document
from src.components.rerankers import BaseReranker, RRFReranker

SEMANTIC_SCHEMA = "semantic_layer"
SEMANTIC_EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class BaseSQLSchemaExtractor(ABC):
    @abstractmethod
    def extract_schema(self) -> dict[str, dict[str, str]]:
        """Возвращает {table_name: {column_name: data_type}}"""
        raise NotImplementedError

    @abstractmethod
    def get_schema_text(self, *args, **kwargs) -> str:
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

    def _fetch(self, query: str, params: tuple | None = None) -> list[tuple]:
        try:
            with self._conn.cursor() as cursor:
                cursor.execute(query, (self._schema,) if params is None else params)
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

    def get_schema_text(self, *args, **kwargs) -> str:
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


class PostgresSQLSemanticContextExtractor(PostgresSQLSchemaExtractor):
    PRIMARY_KEYS_QUERY = """
        SELECT tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = %s
        ORDER BY tc.table_name, kcu.ordinal_position
    """

    def __init__(
        self,
        conn: connection,
        embeddings_model: BaseEmbeddingsModel,
        reranker: BaseReranker,
        schema: str = "public",
        semantic_schema: str = SEMANTIC_SCHEMA,
        candidates_k: int = 8,
        anchors_k: int = 4,
        max_path_length: int = 4,
        examples_k: int = 3,
        metrics_k: int = 3,
        metrics_threshold: float = 0.3,
        values_threshold: float = 0.6,
        inline_values_limit: int = 10,
        verbose: bool = True,
    ):
        super().__init__(conn, schema)
        self._embeddings_model = embeddings_model
        self._reranker = reranker
        self._fusion = RRFReranker()
        self._semantic_schema = semantic_schema
        self._candidates_k = candidates_k
        self._anchors_k = anchors_k
        self._max_path_length = max_path_length
        self._examples_k = examples_k
        self._metrics_k = metrics_k
        self._metrics_threshold = metrics_threshold
        self._values_threshold = values_threshold
        self._inline_values_limit = inline_values_limit
        self._verbose = verbose

    def _embed(self, text: str) -> list[float]:
        return self._embeddings_model.embed_query(text).astype(np.float32).tolist()

    def _log(self, message: str, level: int = 0) -> None:
        if self._verbose:
            print("  " * level + message)

    @staticmethod
    def _table_documents(rows: list[tuple]) -> list[Document]:
        return [
            Document(id=uuid5(NAMESPACE_URL, table_name), text=text, metadata={"table": table_name})
            for table_name, text in rows
        ]

    # 1. Правила диалекта и стиля
    def get_rules(self) -> list[str]:
        rows = self._fetch(f"SELECT text FROM {self._semantic_schema}.rules ORDER BY id", ())
        rules = [row[0] for row in rows]
        self._log(f"[1] Правила ({self._semantic_schema}.rules, берутся все): {len(rules)} шт.")
        return rules

    # 2. Метрики, измерения и глоссарий
    def search_metrics(self, query: str) -> list[dict]:
        embedding = self._embed(query)
        rows = self._fetch(
            f"""
            SELECT kind, name, description, sql, filters, 1 - (embedding <=> %s::vector) AS score
            FROM {self._semantic_schema}.metrics
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, self._metrics_k),
        )
        self._log(
            f"[2] Метрики, измерения, глоссарий (вектор по {self._semantic_schema}.metrics, "
            f"top-{self._metrics_k}, порог {self._metrics_threshold}):"
        )
        for kind, name, _, _, _, score in rows:
            status = "в контекст" if score >= self._metrics_threshold else "отброшено (ниже порога)"
            self._log(f"{name} ({kind}) score={score:.3f} -> {status}", 1)
        return [
            {"kind": kind, "name": name, "description": description, "sql": sql, "filters": filters}
            for kind, name, description, sql, filters, score in rows
            if score >= self._metrics_threshold
        ]

    # 3. Schema linking: гибридный поиск (вектор + полнотекстовый) -> RRF -> cross-encoder
    def search_tables(self, query: str) -> list[str]:
        vector_rows = self._fetch(
            f"""
            SELECT table_name, text FROM {self._semantic_schema}.table_docs
            ORDER BY embedding <=> %s::vector LIMIT %s
            """,
            (self._embed(query), self._candidates_k),
        )
        tokens = re.findall(r"\w+", query.lower())
        fts_rows = []
        if tokens:
            fts_rows = self._fetch(
                f"""
                SELECT table_name, text FROM {self._semantic_schema}.table_docs, to_tsquery('simple', %s) q
                WHERE tsv @@ q ORDER BY ts_rank(tsv, q) DESC LIMIT %s
                """,
                (" | ".join(tokens), self._candidates_k),
            )
        self._log(f"[3] Поиск таблиц (schema linking) по {self._semantic_schema}.table_docs:")
        self._log(f"3.1 Векторный поиск, top-{self._candidates_k}: {[name for name, _ in vector_rows]}", 1)
        self._log(f"3.2 Полнотекстовый поиск, top-{self._candidates_k}: {[name for name, _ in fts_rows]}", 1)
        candidates = self._fusion.rerank(
            query, self._table_documents(vector_rows), self._table_documents(fts_rows), top_k=self._candidates_k
        )
        self._log(f"3.3 Слияние RRF: {[doc.metadata['table'] for doc in candidates]}", 1)
        reranked = self._reranker.rerank(query, candidates, top_k=self._anchors_k)
        anchors = [doc.metadata["table"] for doc in reranked]
        self._log(f"3.4 Реранк cross-encoder, top-{self._anchors_k} (якоря): {anchors}", 1)
        return anchors

    # 4. Расширение найденных таблиц по графу внешних ключей
    def expand_by_graph(self, anchors: list[str]) -> list[str]:
        neighbors: dict[str, set[str]] = {}
        for table, _, foreign_table, _ in self.extract_foreign_keys():
            if table != foreign_table:
                neighbors.setdefault(table, set()).add(foreign_table)
                neighbors.setdefault(foreign_table, set()).add(table)

        def shortest_path(start: str, end: str) -> list[str] | None:
            queue = deque([[start]])
            visited = {start}
            while queue:
                path = queue.popleft()
                if path[-1] == end:
                    return path
                for neighbor in sorted(neighbors.get(path[-1], ())):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append([*path, neighbor])
            return None

        self._log(f"[4] Расширение по графу внешних ключей (макс. длина пути {self._max_path_length}):")
        result = list(anchors)
        for start, end in combinations(anchors, 2):
            path = shortest_path(start, end)
            if path is None:
                self._log(f"{start} - {end}: пути нет", 1)
            elif len(path) - 1 > self._max_path_length:
                self._log(f"{start} - {end}: путь {' -> '.join(path)} слишком длинный, пропущен", 1)
            else:
                added = [table for table in path if table not in result]
                result.extend(added)
                self._log(f"{start} - {end}: путь {' -> '.join(path)}, добавлено: {added}", 1)
        self._log(f"Итоговые таблицы: {result}", 1)
        return result

    # 5. DDL выбранных таблиц с описаниями, PK и FK
    def get_ddl_context(self, tables: list[str]) -> str:
        schema = self.extract_schema()
        primary_keys = set(self._fetch(self.PRIMARY_KEYS_QUERY))
        foreign_keys = {(t, c): f"{ft}.{fc}" for t, c, ft, fc in self.extract_foreign_keys()}
        docs = {
            name: (description, columns)
            for name, description, columns in self._fetch(
                f"""
                SELECT table_name, description, columns FROM {self._semantic_schema}.table_docs
                WHERE table_name = ANY(%s)
                """,
                (tables,),
            )
        }
        blocks = []
        for table in tables:
            description, column_descriptions = docs.get(table, ("", {}))
            lines = [f"{table}: {description}" if description else table]
            for column, data_type in schema.get(table, {}).items():
                flags = ""
                if (table, column) in primary_keys:
                    flags += " PRIMARY KEY"
                if (table, column) in foreign_keys:
                    flags += f" REFERENCES {foreign_keys[(table, column)]}"
                column_description = column_descriptions.get(column)
                lines.append(f"  {column} {data_type}{flags}" + (f" -- {column_description}" if column_description else ""))
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    # 6. Значения категориальных колонок
    def get_inline_values(self, tables: list[str]) -> dict[str, list[str]]:
        """Колонки с малым числом значений: список значений целиком."""
        rows = self._fetch(
            f"""
            SELECT table_name || '.' || column_name, array_agg(value ORDER BY value)
            FROM {self._semantic_schema}.column_values
            WHERE cardinality <= %s AND table_name = ANY(%s)
            GROUP BY table_name, column_name
            ORDER BY 1
            """,
            (self._inline_values_limit, tables),
        )
        inline_values = dict(rows)
        self._log(f"[6.1] Все значения колонок с <= {self._inline_values_limit} вариантами ({self._semantic_schema}.column_values):")
        for column, values in inline_values.items():
            self._log(f"{column}: {values}", 1)
        return inline_values

    def search_values(self, query: str, tables: list[str]) -> list[tuple[str, str, str]]:
        """Возвращает (колонка, значение из БД, упоминание из вопроса) для колонок с большим числом значений."""
        words = [word for word in re.findall(r"\w+", query) if len(word) >= 3]
        mentions = words + [" ".join(pair) for pair in zip(words, words[1:])]
        matches: dict[tuple[str, str], str] = {}
        for mention in mentions:
            embedding = self._embed(mention)
            rows = self._fetch(
                f"""
                SELECT table_name || '.' || column_name, value, 1 - (embedding <=> %s::vector) AS score
                FROM {self._semantic_schema}.column_values
                WHERE cardinality > %s AND table_name = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (embedding, self._inline_values_limit, tables, embedding),
            )
            for column, value, score in rows:
                if score >= self._values_threshold:
                    matches.setdefault((column, value), mention)
        found = [(column, value, mention) for (column, value), mention in matches.items()]
        self._log(f"[6.2] Значения по упоминаниям из вопроса (вектор, порог {self._values_threshold}):")
        for column, value, mention in found:
            self._log(f"'{mention}' -> {column} = {value!r}", 1)
        if not found:
            self._log("ничего не найдено", 1)
        return found

    # 7. Похожие примеры вопрос-SQL
    def search_examples(self, query: str) -> list[tuple[str, str]]:
        examples = self._fetch(
            f"""
            SELECT question, sql FROM {self._semantic_schema}.query_examples
            ORDER BY embedding <=> %s::vector LIMIT %s
            """,
            (self._embed(query), self._examples_k),
        )
        self._log(f"[7] Похожие примеры (вектор по {self._semantic_schema}.query_examples, top-{self._examples_k}):")
        for question, _ in examples:
            self._log(question, 1)
        return examples

    def get_schema_text(self, *args, **kwargs) -> str:
        query = kwargs.get("query") or (args[0] if args else None)
        if not query:
            raise ValueError("PostgresSQLSemanticContextExtractor requires a query")

        self._log(f"=== Сбор контекста для вопроса: {query!r} ===")
        tables = self.expand_by_graph(self.search_tables(query))
        sections = ["Rules:\n" + "\n".join(f"- {rule}" for rule in self.get_rules())]

        metrics = self.search_metrics(query)
        if metrics:
            lines = []
            for item in metrics:
                line = f"- {item['name']} ({item['kind']}): {item['description']}"
                if item["sql"]:
                    line += f". SQL: {item['sql']}"
                if item["filters"]:
                    line += f". Filters: {' AND '.join(item['filters'])}"
                lines.append(line)
            sections.append("Business definitions (use them instead of inventing formulas):\n" + "\n".join(lines))

        sections.append("Tables and columns:\n" + self.get_ddl_context(tables))
        self._log(f"[5] DDL с описаниями, PK и FK для таблиц: {tables}")

        values_lines = [
            f"- {column}: {', '.join(repr(value) for value in values)}"
            for column, values in self.get_inline_values(tables).items()
        ]
        values_lines += [
            f"- {column} contains the value {value!r} (matches '{mention}' from the question)"
            for column, value, mention in self.search_values(query, tables)
        ]
        if values_lines:
            sections.append("Values stored in the database (use them exactly as written):\n" + "\n".join(values_lines))

        examples = self.search_examples(query)
        if examples:
            sections.append(
                "Examples of similar questions with correct SQL:\n"
                + "\n\n".join(f"Question: {question}\nSQL:\n{sql.strip()}" for question, sql in examples)
            )
        context = "\n\n".join(sections) + "\n"
        self._log(f"=== Итоговый контекст для модели ===\n{context}=== Конец контекста ===")
        return context
