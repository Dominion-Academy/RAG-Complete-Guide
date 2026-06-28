from typing import TYPE_CHECKING

from fastembed import TextEmbedding


if TYPE_CHECKING:
    import numpy as np


class FastEmbedEmbeddingsModel:
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
