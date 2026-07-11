from abc import ABC, abstractmethod
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from src.components.embeddings import BaseEmbeddingsModel


class Document(BaseModel):
    id: UUID = Field(description="Идентификатор документа", default_factory=uuid4)
    text: str = Field(description="Текст документа")
    metadata: dict[str, Any] = Field(description="Метаданные документа", default_factory=dict)


class BaseKnowledgeStorage(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query: str, limit: int) -> list[Document]: ...


class QdrantCollectionKnowledgeStorage(BaseKnowledgeStorage):
    def __init__(self, embeddings_model: BaseEmbeddingsModel, client: QdrantClient, collection_name: str):
        self._embeddings_model = embeddings_model
        self._client = client
        self._collection_name = collection_name

    def create_collection_if_not_exists(self) -> Self:
        if self._client.collection_exists(self._collection_name):
            return self
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._embeddings_model.size,
                distance=Distance.DOT,
                on_disk=False,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=500,
                on_disk=False,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=500,
            ),
        )
        return self

    def drop_collection_if_exists(self) -> Self:
        if not self._client.collection_exists(self._collection_name):
            return self
        self._client.delete_collection(collection_name=self._collection_name)
        return self

    def add_documents(self, documents: list[Document]) -> None:
        doc_embeddings = self._embeddings_model.embed_document([doc.text for doc in documents])
        points = [
            PointStruct(
                id=documents[idx].id,
                vector=doc_embeddings[idx].tolist(),
                payload=dict(text=documents[idx].text, **documents[idx].metadata),
            )
            for idx in range(len(documents))
        ]
        self._client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=points,
        )

    def search(self, query: str, limit: int) -> list[Document]:
        query_embedding = self._embeddings_model.embed_query(query)
        query_embedding_list = query_embedding.tolist()
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding_list,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [Document(id=point.id, text=point.payload.pop("text"), metadata=point.payload) for point in response.points]


class SimpleRetriever:
    def __init__(self, storage: BaseKnowledgeStorage):
        self._storage = storage

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        return self._storage.search(query, top_k)
