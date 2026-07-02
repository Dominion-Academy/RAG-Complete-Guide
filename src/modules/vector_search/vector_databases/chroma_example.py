import random

import chromadb
from chromadb.api import CreateCollectionConfiguration
import pandas as pd

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


# 2. Создаём клиента (режим in-memory)
client = chromadb.Client()


# 3. Создаём коллекцию без предустановленной эмбеддинг-функции
collection = client.create_collection(
    name="documents",
    embedding_function=None,
    configuration=CreateCollectionConfiguration(
        hnsw={
            "space": "cosine",
        }
    ),
)

# 4. Добавляем документы в коллекцию
source_variants = ["wiki", "docs", "paper"]
collection.add(
    ids=list(map(str, range(len(documents)))),
    embeddings=doc_embeddings,
    documents=documents,
    metadatas=[{"source": random.choice(source_variants)} for _ in range(len(documents))],
)

# 4. Поиск по текстовому запросу
results = collection.query(query_embeddings=query_embedding, n_results=5)
print(results)
for doc_id, distance, text in zip(results["ids"][0], results["distances"][0], results["documents"][0]):
    similarity = 1 - distance
    print(f"[doc_id={doc_id}, {similarity:.3f}]\n{text[:100]}\n===")
