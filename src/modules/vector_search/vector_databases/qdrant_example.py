import datetime
import random

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

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
client = QdrantClient(":memory:")


# 3. Создаем коллекцию
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=embeddings_model.size,
        distance=Distance.COSINE,  # метрика: COSINE, DOT, EUCLID, MANHATTAN
        on_disk=False,  # хранить в RAM
    ),
    hnsw_config=HnswConfigDiff(
        m=0,  # m=0 отключает HNSW, используется точный поиск
    ),
)


# 4. Подготовка записей для вставки
source_variants = ["wiki", "docs", "paper"]
points = [
    PointStruct(
        id=i,
        vector=doc_embeddings[i].tolist(),
        payload={
            "text": document_text,
            "source": random.choice(source_variants),
            "created_at": datetime.datetime.now(tz=datetime.UTC),
        },
    )
    for i, document_text in enumerate(documents)
]


# 5. Вставка данных
operation_info = client.upsert(
    collection_name="documents",
    wait=True,
    points=points,
)
print(f"Статус загрузки: {operation_info.status}")


# 6. Выполняем поиск
query_embedding_list = query_embedding.tolist()
response = client.query_points(
    collection_name="documents",
    query=query_embedding_list,
    limit=5,  # топ-K результатов
    with_payload=True,  # вернуть метаданные
    with_vectors=False,  # не возвращать векторы (экономия трафика)
)

for point in response.points:
    print(f"[doc_id={point.id}, {point.score:.3f}]\n{point.payload['text'][:100]}\n===")
