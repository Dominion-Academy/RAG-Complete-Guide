from string import Template

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Fusion, FusionQuery, PointStruct, Prefetch, SparseVectorParams, VectorParams

from src.components.embeddings import BaseEmbeddingsModel, BaseSparseEmbeddingsModel
from src.components.knowledge_storage import Document
from src.components.llm import AIAnswer, InputMessage, SyncOpenAILikeLLM


class QdrantFusionRAG:
    SYSTEM_PROMPT = Template(
        "You are a useful assistant. Use the language specified in $lang for your entire response. "
        "Answer the user's question briefly using only contextual data."
    )
    USER_PROMPT = Template("Question: $query\n\nContext (the ONLY source of truth):\n$context")

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model: BaseEmbeddingsModel,
        sparse_model: BaseSparseEmbeddingsModel,
        llm: SyncOpenAILikeLLM,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self._llm = llm

    def create_collection(
        self, dense_vector_params: None | dict = None, sparse_vector_params: None | dict = None, drop_if_exists: bool = False
    ) -> None:
        is_existing = self._client.collection_exists(self._collection_name)
        if is_existing:
            if drop_if_exists:
                self._client.delete_collection(self._collection_name)
            else:
                raise ValueError("Коллекция с таким именем уже существует")

        dense_vector_params = dense_vector_params or dict()
        sparse_vector_params = sparse_vector_params or dict()

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self._dense_model.size,
                    distance=Distance.DOT,
                    **dense_vector_params,
                )
            },
            sparse_vectors_config={"sparse": SparseVectorParams(**sparse_vector_params)},
        )

    def add_documents(self, documents: list[Document], send_batch_size: int = 100) -> None:
        sparse_embeddings = self._sparse_model.embed_document([doc.text for doc in documents])
        dense_embeddings = self._dense_model.embed_document([doc.text for doc in documents])
        points = [
            PointStruct(
                id=doc.id,
                vector={
                    "dense": dense_embedding,
                    "sparse": sparse_embedding,
                },
                payload={"text": doc.text, **doc.metadata},
            )
            for doc, dense_embedding, sparse_embedding in zip(documents, dense_embeddings, sparse_embeddings)
        ]
        for i in range(0, len(points), send_batch_size):
            self._client.upsert(self._collection_name, points[i : i + send_batch_size])

    def _retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        dense_query_embedding = self._dense_model.embed_query(query)
        sparse_query_embedding = self._sparse_model.embed_query(query)
        response = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(
                    query=sparse_query_embedding,
                    using="sparse",
                    limit=top_k,
                ),
                Prefetch(
                    query=dense_query_embedding,
                    using="dense",
                    limit=top_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.DBSF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [Document(id=point.id, text=point.payload.pop("text"), metadata=point.payload) for point in response.points]

    def generate(self, query: str, top_k: int = 10) -> AIAnswer:
        documents = self._retrieve(query, top_k=top_k)

        system_prompt = self.SYSTEM_PROMPT.substitute({"lang": "english"})
        context = "\n\n".join([doc.text for doc in documents])
        user_prompt = self.USER_PROMPT.substitute({"query": query, "context": context})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]
        return self._llm.generate_answer(messages)
