from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.modules.vector_search.extra.data import POINTS


# 1. Загружаем данные и получаем эмбеддинги
embeddings_model = FastEmbedEmbeddingsModel()
doc_embeddings = embeddings_model.embed_document([p["text"] for p in POINTS])


# 2. Подключаемся к базе данных
client = QdrantClient(url="http://localhost:6333")


# 3. Создаем коллекцию
collection_name = "qdrant_filtering_documents"
client.delete_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embeddings_model.size,
        distance=Distance.DOT,
    ),
)


# 4. Подготовка записей для вставки
points = [
    PointStruct(
        id=point["id"],
        vector=doc_embeddings[i].tolist(),
        payload={"text": point["text"], **point["metadata"]},
    )
    for i, point in enumerate(POINTS)
]


# 5. Вставка данных с параллельной загрузкой
operation_info = client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points,
)


# Простые условия
response = client.query_points(
    collection_name=collection_name,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="access_level",
                match=MatchValue(value="public"),
            ),
            FieldCondition(
                key="department",
                match=MatchValue(value="research"),
            ),
        ]
    ),
    limit=3,
    with_payload=True,
    with_vectors=False,
)

print('access_level == "public" AND department == "research"')
for point in response.points:
    print(f"[doc_id={point.id}, {point.score:.3f}]\n{point.payload}\n===")


# Сложные условия

response = client.query_points(
    collection_name=collection_name,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="access_level",
                match=MatchValue(value="public"),
            ),
        ],
        should=[
            FieldCondition(
                key="department",
                match=MatchValue(value="research"),
            ),
            FieldCondition(
                key="department",
                match=MatchValue(value="marketing"),
            ),
        ],
        must_not=[
            FieldCondition(
                key="category",
                match=MatchValue(value="медицина"),
            ),
        ],
    ),
    limit=3,
    with_payload=True,
    with_vectors=False,
)
print()
print(
    '1) access_level == "public"',
    '2) department == "research" OR department == "marketing"',
    '3) category != "медицина"',
    sep="\n",
)
for point in response.points:
    print(f"[doc_id={point.id}, {point.score:.3f}]\n{point.payload}\n===")


# Скроллинг
print()
print("Scrolling")
filters = Filter(
    should=[
        FieldCondition(
            key="department",
            match=MatchValue(value="research"),
        ),
        FieldCondition(
            key="department",
            match=MatchValue(value="marketing"),
        ),
    ],
)
records_1, offset_1 = client.scroll(collection_name=collection_name, scroll_filter=filters, limit=2, with_payload=True)
print(records_1, offset_1, sep="\n")

records_2, offset_2 = client.scroll(
    collection_name=collection_name, scroll_filter=filters, limit=2, with_payload=True, offset=offset_1
)
print(records_2, offset_2, sep="\n")
