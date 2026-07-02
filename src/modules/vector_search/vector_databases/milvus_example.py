import os.path
import random
import shutil

import pandas as pd
from pymilvus import DataType, MilvusClient

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


# 2. Создаём клиента с подключением к локальной базе данных
db_path = "./milvus.db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
client = MilvusClient(uri=db_path)


# 3. Описываем схему и создаем коллекцию
collection_name = "documents"

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embeddings_model.size)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=10_000)
schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=5)

client.create_collection(collection_name=collection_name, schema=schema)

# 4. Создаем индекс без оптимизаций по скалярному произведению
index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")
client.create_index(collection_name=collection_name, index_params=index_params)

# 5. Вставляем данные
source_variants = ["wiki", "docs", "paper"]
data = [
    {"id": idx, "embedding": emb, "text": doc, "source": random.choice(source_variants)}
    for idx, (emb, doc) in enumerate(zip(doc_embeddings, documents))
]
client.insert(collection_name=collection_name, data=data)


# 6. Поиск по текстовому запросу
client.load_collection(collection_name=collection_name)
results = client.search(
    collection_name=collection_name,
    data=[query_embedding],
    limit=5,
    output_fields=["id", "text"],
)
for item in results[0]:
    doc_id = item["id"]
    similarity = item["distance"]
    text = item["entity"]["text"]
    print(f"[doc_id={doc_id}, {similarity:.3f}]\n{text[:100]}\n===")
