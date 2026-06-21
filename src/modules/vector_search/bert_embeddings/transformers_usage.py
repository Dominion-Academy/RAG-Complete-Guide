# !uv add numpy, torch, transformers
import logging
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# Отключение лишних логов
logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Тип для результата измерения (модель, сходства, время)
Result = dict[str, str | float]  # модель


def cosine_similarity_vec(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Векторы должны быть одномерными")
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
    return float((np.dot(vec1, vec2.T) / np.dot(vec1_norm, vec2_norm))[0, 0])


def encode_text(model: AutoModel, tokenizer: AutoTokenizer, text: str, device: torch.device, max_length: int = 128) -> np.ndarray:
    # получаем токены
    tokens_info = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    tokens_info = {k: v.to(device) for k, v in tokens_info.items()}

    # прогоняем токены через модель
    with torch.no_grad():
        outputs = model(**tokens_info)

    # берем выходные веткоры токенов
    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

    # берем матрицу внимания, которая содержит информацию о токенах (1 если есть, 0 если нет)
    attention_mask = tokens_info["attention_mask"]  # (1, seq_len)
    # создаем маску, для того чтобы исключить влияение пустых токенов на усреднение
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

    # сумма по всем векторам, маска исключает пустые токены
    sum_embeddings = torch.sum(last_hidden * mask, dim=1)  # (1, hidden_dim)
    # количество непустых токенов в маске
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)  # (1, hidden_dim)
    # получение усраднения
    mean_pooled = sum_embeddings / sum_mask  # (1, hidden_dim)

    # получаем чистый вектор размерности размера эмбеддинга
    return mean_pooled.cpu().numpy().squeeze()


class SimilarityBenchmark:
    def __init__(self, model_names: list[str], device: torch.device = None) -> None:
        self.model_names = model_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModel.from_pretrained(name).to(self.device)
                model.eval()

                start = time.perf_counter()
                emb_q = encode_text(model, tokenizer, query, self.device)
                emb_p = encode_text(model, tokenizer, positive, self.device)
                emb_n = encode_text(model, tokenizer, negative, self.device)
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


models = [
    "bert-base-uncased",  # классический BERT
    "roberta-base",  # RoBERTa
    "distilbert-base-uncased",  # DistilBERT
    "Nischal2015/sbert_eng_ques",  # SBERT (MiniLM)
]

query = "What is the capital of France?"
positive = "Paris is the capital of France."
negative = "London is the capital of England."

benchmark = SimilarityBenchmark(models)
results = benchmark.run(query, positive, negative)

# Вывод сводной таблицы
print("\nСводка результатов:")
print(f"{'Модель':<40} {'Sim+':>8} {'Sim-':>8} {'Время (с)':>10}")
for r in results:
    r_str = f"{r['model']:<40} {r['similarity_positive']:>8.4f} {r['similarity_negative']:>8.4f} {r['time_seconds']:>10.4f}"
    print(r_str)
