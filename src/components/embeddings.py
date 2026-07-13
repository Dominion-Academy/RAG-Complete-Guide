from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.models import SparseVector


if TYPE_CHECKING:
    import numpy as np


class BaseEmbeddingsModel(ABC):
    @property
    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray: ...

    @abstractmethod
    def embed_document(self, documents: list[str]) -> list[np.ndarray]: ...


class FastEmbedEmbeddingsModel(BaseEmbeddingsModel):
    """Возврашает уже нормализованные по длине(L2) векторы"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model = TextEmbedding(model_name=model_name)

    @property
    def size(self) -> int:
        return self._model.get_embedding_size(self._model.model_name)

    def embed_query(self, query: str) -> np.array:
        return list(self._model.query_embed(query))[0]

    def embed_document(self, documents: list[str]) -> list[np.array]:
        return list(self._model.passage_embed(documents))


class FastEmbedSparseEmbeddingsModel:
    def __init__(self, model_name: str = "Qdrant/bm25", **kwargs) -> None:
        """
        Параметры для моделей:
        - Qdrant/bm25 (k=1.2, b=0.75, avg_len=256)
        """
        self._model = SparseTextEmbedding(model_name=model_name, **kwargs)

    def embed_query(self, query: str) -> SparseVector:
        embedding = list(self._model.query_embed(query))[0]
        return SparseVector(
            values=embedding.values.tolist(),
            indices=embedding.indices.tolist(),
        )

    def embed_document(self, documents: list[str]) -> list[SparseVector]:
        embeddings = list(self._model.passage_embed(documents))
        return [
            SparseVector(
                values=embedding.values.tolist(),
                indices=embedding.indices.tolist(),
            )
            for embedding in embeddings
        ]
