import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVectorParams, VectorParams

from src.components.embeddings import FastEmbedEmbeddingsModel, FastEmbedSparseEmbeddingsModel
from src.components.knowledge_storage import Document
from src.settings import DATASETS_DIR


# 1. Загружаем данные
DATASET_PATH = DATASETS_DIR / "falcon_refined_web" / "data.csv"
df = pd.read_csv(DATASET_PATH, index_col=False)
texts = df["text"].to_list()[:100]

# 2. Переводим тексты в векторные представления
bm25_model = FastEmbedSparseEmbeddingsModel()
dense_model = FastEmbedEmbeddingsModel()

doc_sparse_embeddings = bm25_model.embed_document(texts)
doc_dense_embeddings = dense_model.embed_document(texts)
documents = [Document(text=text) for text in texts]

# 3. Инициализируем клиент
client = QdrantClient(url="http://localhost:6333")

# 4. Создаем коллекцию
collection_name = "hybrid_documents"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": VectorParams(
            size=dense_model.size,
            distance=Distance.DOT,
            on_disk=False,
        )
    },
    sparse_vectors_config={"bm2_sparse": SparseVectorParams(modifier=Modifier.IDF)},
)

# 5. Добавляем точки в коллекцию
points = [
    PointStruct(
        id=doc.id,
        vector={
            "dense": dense_embedding,
            "bm2_sparse": sparse_embedding,
        },
        payload={"text": doc.text, **doc.metadata},
    )
    for doc, dense_embedding, sparse_embedding in zip(documents, doc_dense_embeddings, doc_sparse_embeddings)
]
client.upsert(collection_name, points)

# 6. Добавляем документы в хранилище
SENDING_BATCH = 100
for i in range(1, len(documents), SENDING_BATCH):
    client.upsert(collection_name, points[i : i + SENDING_BATCH])


# 7. Ищем релевантные документы
query = "What is the Berry Export Summary 2028 and what is its purpose?"
query_dense_embedding = dense_model.embed_query(query)
query_sparse_embedding = bm25_model.embed_query(query)
dense_response = client.query_points(
    collection_name=collection_name,
    query=query_dense_embedding,
    using="dense",
    limit=3,
)
sparse_response = client.query_points(
    collection_name=collection_name,
    query=query_sparse_embedding,
    using="bm2_sparse",
    limit=3,
)

all_points = [*dense_response.points, *sparse_response.points]
unique_doc_ids = {p.id for p in points}
unique_points = [p for p in all_points if p.id in unique_doc_ids]

for point in unique_points:
    print(point.payload["text"][:200])
    print("======")
