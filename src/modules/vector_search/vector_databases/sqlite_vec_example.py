import sqlite3

import numpy as np
import pandas as pd
import sqlite_vec

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.settings import DATASETS_DIR


# Загружаем данные
DATASET_PATH = DATASETS_DIR / "falcon_refined_web" / "data.csv"
df = pd.read_csv(DATASET_PATH, index_col=False)
documents = df["text"].to_list()[:1000]

# Инициализируем эмбеддинговую модель
embeddings_model = FastEmbedEmbeddingsModel()

# Получаем эмбеддинги документов и запроса
doc_embeddings = embeddings_model.embed_document(documents)
query_embedding = embeddings_model.embed_query("What is the Berry Export Summary 2028 and what is its purpose?")

# Создаем базу данных SQLite и подключаем расширение
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Создаем таблицу с метаданными
db.execute("""
    CREATE TABLE documents (
        doc_id INTEGER PRIMARY KEY,
        text   TEXT NOT NULL
    )
""")

# Создаем виртуальную таблицу для векторов (по умолчанию float32)
db.execute(f"""
    CREATE VIRTUAL TABLE vec_index USING vec0(
        doc_id INTEGER PRIMARY KEY,
        embedding float[{embeddings_model.size}]
    )
""")

# Вставляем метаданные (идентификатор и текст)
db.executemany("INSERT INTO documents (doc_id, text) VALUES (?, ?)", [(idx, text) for idx, text in enumerate(documents)])

# Вставляем эмбеддинги (идентификатор и вектор)
db.executemany(
    "INSERT INTO vec_index (doc_id, embedding) VALUES (?, ?)",
    [(i, emb.astype(np.float32).tobytes()) for i, emb in enumerate(doc_embeddings)],
)
db.commit()

# Выполняем поиск
query_blob = query_embedding.astype(np.float32).tobytes()
results = db.execute(
    """
    SELECT
        d.doc_id,
        d.text,
        1 - vec_distance_cosine(v.embedding, ?) AS cosine_similarity
    FROM vec_index v
    JOIN documents d ON d.doc_id = v.doc_id
    ORDER BY cosine_similarity desc    
	LIMIT 5
""",
    (query_blob,),
).fetchall()

for doc_id, text, score in results:
    print(f"[doc_id={doc_id}, {score:.3f}]\n{text[:100]}\n===")
