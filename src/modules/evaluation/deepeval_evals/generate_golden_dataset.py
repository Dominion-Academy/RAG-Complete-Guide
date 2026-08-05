import asyncio
import os

from deepeval.dataset import EvaluationDataset
from deepeval.models import DeepEvalBaseEmbeddingModel, LocalModel
from deepeval.models.llms.local_model import retry_local
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.settings import DATASETS_DIR, settings


DATASET_DIR = DATASETS_DIR / "ai_helpdesk" / "data"
files = os.listdir(DATASET_DIR)
file_paths = [str(os.path.join(DATASET_DIR, file)) for file in files]


MODEL = LocalModel(
    model=settings.llm.MODEL,
    api_key=settings.llm.API_KEY,
    base_url=settings.llm.BASE_URL,
)


class FastEmbedEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,  # игнорируется, оставлен для совместимости
        base_url: str | None = None,  # игнорируется
        generation_kwargs: dict | None = None,  # игнорируется
        **kwargs,
    ):
        self.model_name = model or "sentence-transformers/all-MiniLM-L6-v2"
        self.fastembed_kwargs = kwargs
        self._model = None
        super().__init__(self.model_name)

    def load_model(self) -> FastEmbedEmbeddingsModel:
        if self._model is None:
            self._model = FastEmbedEmbeddingsModel(model_name=self.model_name, **self.fastembed_kwargs)
        return self._model

    @retry_local
    def embed_text(self, text: str) -> list[float]:
        model = self.load_model()
        embedding = model.embed_query(text)
        return embedding.tolist()  # np.ndarray -> List[float]

    @retry_local
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self.load_model()
        embeddings = model.embed_document(texts)
        return [emb.tolist() for emb in embeddings]

    @retry_local
    async def a_embed_text(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    @retry_local
    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)

    def get_model_name(self) -> str:
        """Возвращает имя модели."""
        return f"{self.model_name} (FastEmbed)"


goldens = Synthesizer(model=MODEL, async_mode=False).generate_goldens_from_docs(
    document_paths=file_paths,
    include_expected_output=True,
    context_construction_config=ContextConstructionConfig(
        embedder=FastEmbedEmbeddingModel(model="nomic-ai/nomic-embed-text-v1.5-Q"),
        critic_model=MODEL,
        encoding="utf-8",
    ),
)
print(goldens)

dataset = EvaluationDataset(goldens=goldens)
dataset.save_as(file_type="json", directory="./dataset", file_name="generated_golden_dataset", include_test_cases=True)
