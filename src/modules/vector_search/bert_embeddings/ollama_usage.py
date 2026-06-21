import time

import numpy as np
import ollama


# Тип для результата измерения (модель, сходства, время)
Result = dict[str, str | float]


def cosine_similarity_vec(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычисляет косинусную близость между двумя одномерными векторами."""
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Векторы должны быть одномерными")
    # Приводим к форме (1, dim) для единообразия
    v1 = vec1.reshape(1, -1)
    v2 = vec2.reshape(1, -1)
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    return float((np.dot(v1, v2.T) / np.dot(norm1, norm2))[0, 0])


class SimilarityBenchmark:
    def __init__(self, model_names: list[str]) -> None:
        self.model_names = model_names
        self.results: list[Result] = []

    def run(self, query: str, positive: str, negative: str) -> list[Result]:
        self.results.clear()
        print(f"Запрос: {query}")
        print(f"Позитив: {positive}")
        print(f"Негатив: {negative}\n")

        for name in self.model_names:
            print(f"Обработка модели: {name}")
            try:
                start = time.perf_counter()

                # Отправляем все три текста одним запросом — эмбеддинги вернутся в том же порядке
                response = ollama.embed(model=name, input=[query, positive, negative])
                embeddings = response["embeddings"]  # список из трёх numpy-подобных массивов

                # Преобразуем в numpy-массивы (ollama возвращает list of list)
                emb_q = np.array(embeddings[0], dtype=np.float32)
                emb_p = np.array(embeddings[1], dtype=np.float32)
                emb_n = np.array(embeddings[2], dtype=np.float32)

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
    "jina/jina-embeddings-v2-base-en",
    "all-minilm:v2",
    "nomic-embed-text",
]

query = "What is the capital of France?"
positive = "Paris is the capital of France."
negative = "London is the capital of England."

benchmark = SimilarityBenchmark(models)
results = benchmark.run(query, positive, negative)

print("\nСводка результатов:")
print(f"{'Модель':<40} {'Sim+':>8} {'Sim-':>8} {'Время (с)':>10}")
for r in results:
    print(f"{r['model']:<40} {r['similarity_positive']:>8.4f} {r['similarity_negative']:>8.4f} {r['time_seconds']:>10.4f}")
