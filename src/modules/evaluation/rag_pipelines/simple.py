import os
from string import Template

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.components.knowledge_storage import Document
from src.components.llm import AIAnswer, InputMessage, SyncOpenAILikeLLM
from src.components.text_splitters import recursive_split
from src.settings import DATASETS_DIR, settings


DATASET_DIR = DATASETS_DIR / "ai_helpdesk" / "data"


def split_by_h1(text: str) -> list[str]:
    parts = text.split("\n# ")
    return parts


class SimplePipeline:
    SYSTEM_PROMPT = Template(
        "You are a useful assistant. Use language of the user question for your entire response. "
        "Answer the user's question briefly using only contextual data."
        "\n\nContext (the ONLY source of truth):\n$context"
    )

    def __init__(self, llm: SyncOpenAILikeLLM, client: QdrantClient, dense_model: FastEmbedEmbeddingsModel):
        self._llm = llm
        self._client = client
        self._dense_model = dense_model
        self._collection_name = "simple_rag_eval"

    def load_data(self) -> None:
        # Каждый текст начинается с H1
        metadata_list = []
        files = os.listdir(DATASET_DIR)
        for file in files:
            file_path = os.path.join(DATASET_DIR, file)
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
            text = ""
            for line in lines:
                if line.startswith("# "):
                    if text != "":
                        metadata_list.append(dict(text=text, file=file))
                        text = ""
                    text += line
                else:
                    text += line
            if text != "":
                metadata_list.append(dict(text=text, file=file))

        # Чанкинг
        documents: list[Document] = []
        for metadata in metadata_list:
            text = metadata["text"]
            heading = text.split("\n", maxsplit=1)[0][2:]
            chunks = recursive_split(text, max_chars=512, overlap=128)
            documents.extend([Document(text=chunk, metadata=dict(file=metadata["file"], heading=heading)) for chunk in chunks])

        # Создаем колллекцию
        collection_name = "simple_rag_eval"
        is_existing = self._client.collection_exists(collection_name)
        if is_existing:
            self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._dense_model.size,
                distance=Distance.DOT,
            ),
        )

        # Добавляем документы
        send_batch_size = 100
        dense_embeddings = self._dense_model.embed_document([doc.text for doc in documents])
        points = [
            PointStruct(
                id=doc.id,
                vector=dense_embedding,
                payload={"text": doc.text, **doc.metadata},
            )
            for doc, dense_embedding in zip(documents, dense_embeddings)
        ]
        for i in range(0, len(points), send_batch_size):
            self._client.upsert(self._collection_name, points[i : i + send_batch_size])

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        query_embedding = self._dense_model.embed_query(query)

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [Document(id=point.id, text=point.payload.pop("text"), metadata=point.payload) for point in response.points]

    def generate(self, query: str, top_k: int = 10) -> AIAnswer:
        documents = self.retrieve(query, top_k=top_k)
        context = "\n\n".join([doc.text for doc in documents])
        system_prompt = self.SYSTEM_PROMPT.substitute({"context": context})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=query)]
        return self._llm.generate_answer(messages)


llm = SyncOpenAILikeLLM(
    base_url=settings.llm.BASE_URL,
    api_key=settings.llm.API_KEY,
    model=settings.llm.MODEL,
    common_parameters={"temperature": 0},
)
client = QdrantClient(url="http://localhost:6333")
dense_model = FastEmbedEmbeddingsModel(model_name="nomic-ai/nomic-embed-text-v1.5-Q")

pipeline = SimplePipeline(llm, client, dense_model)

if __name__ == "__main__":
    # pipeline.load_data()
    question = "Как кандидат без зимбабвийского номера может откликнуться на вакансию и получить подтверждение отклика?"
    docs = pipeline.retrieve(question)
    for doc in docs:
        print(doc.metadata, doc.text, sep="\n", end="\n\n")
    answer = pipeline.generate(question)
    print(answer)
