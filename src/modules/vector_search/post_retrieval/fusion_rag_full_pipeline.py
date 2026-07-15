import pandas as pd
from qdrant_client import QdrantClient

from src.architectures.fusion_rag import QdrantFusionRAG
from src.components.embeddings import FastEmbedEmbeddingsModel, FastEmbedSparseEmbeddingsModel
from src.components.knowledge_storage import Document
from src.components.llm import SyncOpenAILikeLLM
from src.components.text_splitters import recursive_split
from src.settings import DATASETS_DIR, settings


# 1. Загружаем данные и делим тексты на чанки
DATASET_PATH = DATASETS_DIR / "falcon_refined_web" / "data.csv"
df = pd.read_csv(DATASET_PATH, index_col=False)
texts = df["text"].to_list()[:100]
documents: list[Document] = []
for idx, text in enumerate(texts):
    chunks = recursive_split(text, max_chars=1024, overlap=256)
    documents.extend([Document(text=chunk, metadata=dict(doc_id=idx, number=number)) for number, chunk in enumerate(chunks)])

# 2. Подготавливаем компоненты
collection_name = "fusion_documents_full"
splade_model = FastEmbedSparseEmbeddingsModel(model_name="prithivida/Splade_PP_en_v1")
dense_model = FastEmbedEmbeddingsModel()
client = QdrantClient(url="http://localhost:6333")
llm = SyncOpenAILikeLLM(
    base_url=settings.llm.BASE_URL,
    api_key=settings.llm.API_KEY,
    model=settings.llm.MODEL,
    common_parameters={"temperature": 0},
)

# 3. Инициализируем RAG-пайплайн
rag = QdrantFusionRAG(client=client, collection_name=collection_name, dense_model=dense_model, sparse_model=splade_model, llm=llm)

# 4. Создаем коллекцию
rag.create_collection(drop_if_exists=True)

# 5. Добавляем в индекс документы
rag.add_documents(documents)

# 6. Запускаем генерацию
query = "What is the Berry Export Summary 2028 and what is its purpose?"
answer = rag.generate(query, top_k=3)
print(f"\nAnswer:\n{answer.content}\nInput tokens: {answer.input_tokens}\nOutput tokens: {answer.output_tokens}")
