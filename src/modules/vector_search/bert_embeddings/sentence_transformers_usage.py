# !uv add numpy sentence-transformers
import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer


# Отключение лишних логов
logging.basicConfig(level=logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Тип для результата измерения (модель, сходства, время)
Result = dict[str, str | float]


def cosine_similarity_vec(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Векторы должны быть одномерными")
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
    return float((np.dot(vec1, vec2.T) / np.dot(vec1_norm, vec2_norm))[0, 0])


def encode_text(model: SentenceTransformer, text: str) -> np.ndarray:
    # не нормируем, чтобы потом посчитать косинус вручную
    return model.encode(text, normalize_embeddings=False)


class SimilarityBenchmark:
    def __init__(self, model_names: list[str], device: str = None) -> None:
        self.model_names = model_names
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.results: list[Result] = []

    def run(self, query: str, positive: str, negative: str) -> list[Result]:
        """
        Запускает бенчмарк для всех моделей.

        Returns:
            List[Result]: список словарей с полями:
                'model': имя модели,
                'similarity_positive': сходство с позитивом,
                'similarity_negative': сходство с негативом,
                'time_seconds': время кодирования трёх текстов
        """
        self.results.clear()
        print(f"Запрос: {query}")
        print(f"Позитив: {positive}")
        print(f"Негатив: {negative}\n")

        for name in self.model_names:
            print(f"Обработка модели: {name}")
            try:
                # Загружаем модель sentence-transformers на указанное устройство
                model = SentenceTransformer(name, device=self.device)
                model.eval()

                start = time.perf_counter()
                emb_q = encode_text(model, query)
                emb_p = encode_text(model, positive)
                emb_n = encode_text(model, negative)
                elapsed = time.perf_counter() - start

                sim_pos = cosine_similarity_vec(emb_q, emb_p)
                sim_neg = cosine_similarity_vec(emb_q, emb_n)

                result = {
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


# Современные продвинутые модели sentence-transformers
models = [
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-MiniLM-L6-v2",
    "Nischal2015/sbert_eng_ques",
]

query = "What is the capital of France?"
positive = "Paris is the capital of France."
negative = "London is the capital of England."

benchmark_1 = SimilarityBenchmark(models)
results_1 = benchmark_1.run(query, positive, negative)

benchmark_2 = SimilarityBenchmark(["intfloat/multilingual-e5-small"])
results_2 = benchmark_2.run("query: " + query, "passage: " + positive, "passage: " + negative)

benchmark_3 = SimilarityBenchmark(["intfloat/multilingual-e5-large-instruct"])
results_3 = benchmark_3.run(
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " + query, positive, negative
)

results = results_1 + results_2 + results_3

# Вывод сводной таблицы
print("\nСводка результатов:")
print(f"{'Модель':<40} {'Sim+':>8} {'Sim-':>8} {'Время (с)':>10}")
for r in results:
    r_str = f"{r['model']:<40} {r['similarity_positive']:>8.4f} {r['similarity_negative']:>8.4f} {r['time_seconds']:>10.4f}"
    print(r_str)
