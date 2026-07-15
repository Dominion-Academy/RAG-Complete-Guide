import pandas as pd
from qdrant_client import QdrantClient

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.components.knowledge_storage import Document, QdrantCollectionKnowledgeStorage
from src.components.text_splitters import split_by_words
from src.settings import DATASETS_DIR


# 1. Загружаем данные
DATASET_PATH = DATASETS_DIR / "falcon_refined_web" / "data.csv"
df = pd.read_csv(DATASET_PATH, index_col=False)
texts = df["text"].to_list()[:100]

# 2. Создаем хранилище
knowledge_storage = QdrantCollectionKnowledgeStorage(
    embeddings_model=FastEmbedEmbeddingsModel(),
    client=QdrantClient(url="http://localhost:6333"),
    collection_name="canonical_rag_documents",
)
knowledge_storage.drop_collection_if_exists()
knowledge_storage.create_collection_if_not_exists()

# 3. Разбиваем тексты на чанки
documents: list[Document] = []
for idx, text in enumerate(texts):
    chunks = split_by_words(text)
    documents.extend([Document(text=chunk, metadata=dict(doc_id=idx, number=number)) for number, chunk in enumerate(chunks)])

# 4. Добавляем документы в хранилище
SENDING_BATCH = 100
for i in range(0, len(documents), SENDING_BATCH):
    knowledge_storage.add_documents(documents[i : i + SENDING_BATCH])
print("Готово")
