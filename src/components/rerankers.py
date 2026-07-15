from abc import ABC, abstractmethod
from uuid import UUID

from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.components.knowledge_storage import Document
from src.components.utils import deduplicate_documents


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, *document_groups: list[Document], top_k: int = 5) -> list[Document]: ...


class FastEmbedReranker(BaseReranker):
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, *document_groups: list[Document], top_k: int = 5) -> list[Document]:
        unique_documents = deduplicate_documents(*document_groups)
        scores = self._model.rerank(query, [doc.text for doc in unique_documents])
        return [doc for _, doc in sorted(zip(scores, unique_documents), key=lambda x: x[0], reverse=True)][:top_k]


class RRFReranker(BaseReranker):
    def __init__(self, k: int = 60):
        self._k = k

    def rerank(self, query: str, *document_groups: list[Document], top_k: int = 5) -> list[Document]:
        document_score_map: dict[UUID:float] = {}
        document_map: dict[UUID:Document] = dict()
        for documents in document_groups:
            for rank, doc in enumerate(documents):
                document_map[doc.id] = doc
                document_score_map[doc.id] = document_score_map.get(doc.id, 0) + 1 / (rank + 1 + self._k)

        doc_scores = sorted(document_score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [document_map[doc_id] for doc_id, _ in doc_scores]
