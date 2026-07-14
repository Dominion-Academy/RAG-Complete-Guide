import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Modifier, PointStruct, SparseVectorParams

from src.components.embeddings import FastEmbedSparseEmbeddingsModel
from src.components.knowledge_storage import Document
from src.settings import DATASETS_DIR


# 1. Загружаем данные
DATASET_PATH = DATASETS_DIR / "falcon_refined_web" / "data.csv"
df = pd.read_csv(DATASET_PATH, index_col=False)
texts = df["text"].to_list()[:100]

# 2. Переводим тексты в разреженные векторные представления
bm25_model = FastEmbedSparseEmbeddingsModel()
doc_embeddings = bm25_model.embed_document(texts)
documents = [Document(text=text) for text in texts]

# 3. Инициализируем клиент
client = QdrantClient(url="http://localhost:6333")

# 4. Создаем коллекцию
collection_name = "sparse_documents"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={},
    sparse_vectors_config={"bm2_sparse": SparseVectorParams(modifier=Modifier.IDF)},
)

# 5. Добавляем точки в коллекцию
points = [
    PointStruct(
        id=doc.id,
        vector={
            "bm2_sparse": doc_embedding,
        },
        payload={"text": doc.text, **doc.metadata},
    )
    for doc, doc_embedding in zip(documents, doc_embeddings)
]
client.upsert(collection_name, points)

# 6. Добавляем документы в хранилище
SENDING_BATCH = 100
for i in range(0, len(documents), SENDING_BATCH):
    client.upsert(collection_name, points[i : i + SENDING_BATCH])


# 7. Ищем релевантные документы
query = "What is the Berry Export Summary 2028 and what is its purpose?"
query_embedding = bm25_model.embed_query(query)
response = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    using="bm2_sparse",
    limit=3,
)

for point in response.points:
    print(point.payload["text"][:200])
    print("======")
