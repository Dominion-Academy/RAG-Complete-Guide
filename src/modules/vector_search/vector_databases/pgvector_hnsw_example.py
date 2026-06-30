import numpy as np
import pandas as pd
import psycopg2

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.settings import DATASETS_DIR


# 1. Загружаем данные и получаем эмбеддинги
DATASET_PATH = DATASETS_DIR / "falcon_refined_web" / "data.csv"
df = pd.read_csv(DATASET_PATH, index_col=False)
documents = df["text"].to_list()[:1000]
query = "What is the Berry Export Summary 2028 and what is its purpose?"

embeddings_model = FastEmbedEmbeddingsModel()
doc_embeddings = embeddings_model.embed_document(documents)
query_embedding = embeddings_model.embed_query(query)


# 2. Подключаемся к базе данных
conn = psycopg2.connect(host="localhost", port=5432, dbname="postgres", user="postgres", password="password")
conn.autocommit = True
cur = conn.cursor()

# Включаем расширение (если ещё не включено)
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# 3. Создаём таблицу
cur.execute("DROP TABLE IF EXISTS documents")
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding vector({embeddings_model.size})
    )
""")
cur.execute("""
    CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
conn.commit()

# 4. Вставляем данные по одному вектору
for i, (text, emb) in enumerate(zip(documents, doc_embeddings)):
    cur.execute(
        "INSERT INTO documents (id, text, embedding) VALUES (%s, %s, %s::vector)", (i, text, emb.astype(np.float32).tolist())
    )


# 5. Выполняем поиск
query_embedding_list = query_embedding.tolist()
cur.execute(
    """
    SELECT id, text, 1 - (embedding <=> %s::vector) AS score
    FROM documents
    ORDER BY 1 - (embedding <=> %s::vector) desc
    LIMIT 5
""",
    (query_embedding_list, query_embedding_list),
)

for row in cur.fetchall():
    print(f"[doc_id={row[0]}, distance={row[2]:.3f}]\n{row[1][:100]}\n===")

cur.close()
conn.close()
