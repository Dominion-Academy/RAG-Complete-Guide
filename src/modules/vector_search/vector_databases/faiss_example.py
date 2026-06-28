import faiss
import numpy as np
import pandas as pd

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.settings import DATASETS_DIR


# Загружаем данные
DATASET_PATH = DATASETS_DIR / "falcon_refined_web" / "data.csv"
df = pd.read_csv(DATASET_PATH, index_col=False)
documents = df["text"].to_list()[:1000]

# Инициализируем эмбеддинговую модель
embeddings_model = FastEmbedEmbeddingsModel()

# Создаем поисковый индекс FAISS (без оптимизаций)
index = faiss.IndexFlatIP(embeddings_model.size)

# Получаем эмбеддинги документов и запроса
doc_embeddings = embeddings_model.embed_document(documents)
query_embedding = embeddings_model.embed_query("What is the Berry Export Summary 2028 and what is its purpose?")

# Загружаем эмбеддинги документов в индекс
index.add(np.array(doc_embeddings))

# Смотрим на результаты
scores, indices = index.search(query_embedding.reshape(1, -1), k=5)
for score, idx in zip(scores[0], indices[0]):
    print(f"[doc_id={idx}, {score:.3f}]\n{documents[idx][:100]}\n===")
