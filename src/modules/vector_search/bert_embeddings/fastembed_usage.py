import logging
import time

from fastembed import TextEmbedding
import numpy as np


logging.basicConfig(level=logging.ERROR)
logging.getLogger("fastembed").setLevel(logging.ERROR)

# Тип для результата измерения (модель, сходства, время)
Result = dict[str, str | float]


def cosine_similarity_vec(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Векторы должны быть одномерными")
    # Приводим к форме (1, dim) для единообразия
    v1 = vec1.reshape(1, -1)
    v2 = vec2.reshape(1, -1)
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    return float((np.dot(v1, v2.T) / np.dot(norm1, norm2))[0, 0])


class SimilarityBenchmark:
    def __init__(self, model_names: list[str], device: str = "auto") -> None:
        self.model_names = model_names
        self.device = device
        self.results: list[Result] = []

    def run(self, query: str, positive: str, negative: str) -> list[Result]:
        self.results.clear()
        print(f"Запрос: {query}")
        print(f"Позитив: {positive}")
        print(f"Негатив: {negative}\n")

        for name in self.model_names:
            print(f"Обработка модели: {name}")
            try:
                model = TextEmbedding(
                    model_name=name,
                    device=self.device if self.device != "auto" else None,
                )
                start = time.perf_counter()

                # Для запроса используем query_embed, для документов – passage_embed
                # fastembed сам добавляет нужные префиксы (для BGE, E5, GTE и др.)
                emb_q = next(model.query_embed([query]))  # numpy array
                emb_p, emb_n = list(model.passage_embed([positive, negative]))

                elapsed = time.perf_counter() - start

                sim_pos = cosine_similarity_vec(emb_q, emb_p)
                sim_neg = cosine_similarity_vec(emb_q, emb_n)

                result: Result = {
                    "model": name,
                    "similarity_positive": sim_pos,
                    "similarity_negative": sim_neg,
                    "time_seconds": elapsed,
                }
                self.results.append(result)

                print(f"  Близость к позитивному примеру: {sim_pos:.4f}")
                print(f"  Близость к негативному примеру: {sim_neg:.4f}")
                print(f"  Время в секундах: {elapsed:.4f} с\n")

            except Exception as e:
                print(f"  Ошибка: {e}\n")

        return self.results


models = [
    "BAAI/bge-small-en",
    "snowflake/snowflake-arctic-embed-s",
    "sentence-transformers/all-MiniLM-L6-v2",
    "thenlper/gte-base",
]

query = "What is the capital of France?"
positive = "Paris is the capital of France."
negative = "London is the capital of England."

# Один бенчмарк для всех моделей — префиксы добавляются автоматически
benchmark = SimilarityBenchmark(models, device="auto")
results = benchmark.run(query, positive, negative)

print("\nСводка результатов:")
print(f"{'Модель':<40} {'Sim+':>8} {'Sim-':>8} {'Время (с)':>10}")
for r in results:
    print(f"{r['model']:<40} {r['similarity_positive']:>8.4f} {r['similarity_negative']:>8.4f} {r['time_seconds']:>10.4f}")
