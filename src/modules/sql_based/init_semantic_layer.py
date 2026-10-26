import json
from pathlib import Path

import numpy as np
import psycopg2
import yaml

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.components.sql_schema_extractors import SEMANTIC_EMBEDDINGS_MODEL, SEMANTIC_SCHEMA
from src.settings import settings

CONFIG_PATH = Path(__file__).parent / "semantic_layer.yaml"
INLINE_VALUES_LIMIT = 10

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
embeddings_model = FastEmbedEmbeddingsModel(SEMANTIC_EMBEDDINGS_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return [embedding.astype(np.float32).tolist() for embedding in embeddings_model.embed_document(texts)]


conn = psycopg2.connect(**settings.postgres.connect_kwargs())
cur = conn.cursor()

try:
    # 1. Проверяем SQL в примерах: неверный пример вредит сильнее, чем его отсутствие
    for example in config["examples"]:
        try:
            cur.execute(example["sql"])
            cur.fetchall()
        except Exception as error:
            raise ValueError(f"Невалидный пример '{example['question']}': {error}") from error
        finally:
            conn.rollback()

    # 2. Создаём схему семантического слоя (отдельно от public, чтобы она не попала в схему для LLM)
    size = embeddings_model.size
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    cur.execute(f"DROP SCHEMA IF EXISTS {SEMANTIC_SCHEMA} CASCADE")
    cur.execute(f"CREATE SCHEMA {SEMANTIC_SCHEMA}")
    cur.execute(f"""
        CREATE TABLE {SEMANTIC_SCHEMA}.rules (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL
        );
        CREATE TABLE {SEMANTIC_SCHEMA}.table_docs (
            table_name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            columns JSONB NOT NULL,
            text TEXT NOT NULL,
            embedding vector({size}) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
        );
        CREATE INDEX table_docs_embedding_hnsw ON {SEMANTIC_SCHEMA}.table_docs
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        CREATE INDEX table_docs_tsv ON {SEMANTIC_SCHEMA}.table_docs USING gin (tsv);
        CREATE TABLE {SEMANTIC_SCHEMA}.metrics (
            id SERIAL PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('metric', 'dimension', 'glossary')),
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            sql TEXT,
            filters TEXT[] NOT NULL DEFAULT '{{}}',
            synonyms TEXT[] NOT NULL DEFAULT '{{}}',
            embedding vector({size}) NOT NULL
        );
        CREATE INDEX metrics_embedding_hnsw ON {SEMANTIC_SCHEMA}.metrics
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        CREATE TABLE {SEMANTIC_SCHEMA}.query_examples (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            sql TEXT NOT NULL,
            embedding vector({size}) NOT NULL
        );
        CREATE INDEX query_examples_embedding_hnsw ON {SEMANTIC_SCHEMA}.query_examples
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        CREATE TABLE {SEMANTIC_SCHEMA}.column_values (
            id SERIAL PRIMARY KEY,
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            value TEXT NOT NULL,
            cardinality INTEGER NOT NULL,
            embedding vector({size}) NOT NULL
        );
        CREATE INDEX column_values_embedding_hnsw ON {SEMANTIC_SCHEMA}.column_values
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        CREATE INDEX column_values_trgm ON {SEMANTIC_SCHEMA}.column_values USING gin (value gin_trgm_ops);
    """)

    # 3. Правила
    for rule in config["rules"]:
        cur.execute(f"INSERT INTO {SEMANTIC_SCHEMA}.rules (text) VALUES (%s)", (rule,))

    # 4. Описания таблиц: документ вида "Таблица / Описание / Колонки / Синонимы"
    names = list(config["tables"])
    texts = []
    for name, table in config["tables"].items():
        columns = ", ".join(f"{column} ({description})" for column, description in table["columns"].items())
        texts.append(
            f"Таблица: {name}\nОписание: {table['description']}\nКолонки: {columns}\n"
            f"Синонимы: {', '.join(table['synonyms'])}"
        )
    for name, text, embedding in zip(names, texts, embed(texts)):
        table = config["tables"][name]
        cur.execute(
            f"""
            INSERT INTO {SEMANTIC_SCHEMA}.table_docs (table_name, description, columns, text, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            (name, table["description"], json.dumps(table["columns"], ensure_ascii=False), text, embedding),
        )

    # 5. Метрики, измерения и глоссарий
    items = (
        [("metric", item["name"], item["description"], item) for item in config["metrics"]]
        + [("dimension", item["name"], item["description"], item) for item in config["dimensions"]]
        + [("glossary", item["term"], item["definition"], item) for item in config["glossary"]]
    )
    item_texts = [f"{name}: {description}. {', '.join(item['synonyms'])}" for _, name, description, item in items]
    for (kind, name, description, item), embedding in zip(items, embed(item_texts)):
        cur.execute(
            f"""
            INSERT INTO {SEMANTIC_SCHEMA}.metrics (kind, name, description, sql, filters, synonyms, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            """,
            (kind, name, description, item.get("sql"), item.get("filters", []), item["synonyms"], embedding),
        )

    # 6. Примеры: индексируем текст вопроса, SQL храним в отдельном поле
    questions = [example["question"] for example in config["examples"]]
    for example, embedding in zip(config["examples"], embed(questions)):
        cur.execute(
            f"INSERT INTO {SEMANTIC_SCHEMA}.query_examples (question, sql, embedding) VALUES (%s, %s, %s::vector)",
            (example["question"], example["sql"].strip(), embedding),
        )

    # 7. Значения категориальных колонок берём прямо из данных
    for table_column in config["value_columns"]:
        table, column = table_column.split(".")
        cur.execute(f"SELECT DISTINCT {column}::text FROM {table} WHERE {column} IS NOT NULL ORDER BY 1")
        values = [row[0] for row in cur.fetchall()]
        for value, embedding in zip(values, embed(values)):
            cur.execute(
                f"""
                INSERT INTO {SEMANTIC_SCHEMA}.column_values (table_name, column_name, value, cardinality, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (table, column, value, len(values), embedding),
            )

    conn.commit()
    for table in ["rules", "table_docs", "metrics", "query_examples", "column_values"]:
        cur.execute(f"SELECT COUNT(*) FROM {SEMANTIC_SCHEMA}.{table}")
        print(f"{table}: {cur.fetchone()[0]}")
finally:
    cur.close()
    conn.close()
