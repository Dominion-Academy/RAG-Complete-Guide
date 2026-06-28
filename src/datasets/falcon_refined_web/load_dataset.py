import pandas as pd

from datasets import load_dataset
from src.settings import DATASETS_DIR, settings


# Загрузка датасета
rag_dataset = load_dataset("neural-bridge/rag-dataset-12000", split="train", token=settings.HF_TOKEN)

# Извлечение данных
contexts = rag_dataset["context"]
questions = rag_dataset["question"]
answers = rag_dataset["answer"]

# Формирование data.csv (doc_id, text)
data_df = pd.DataFrame({"doc_id": list(range(len(contexts))), "text": list(contexts)})
data_df.to_csv(DATASETS_DIR / "falcon_refined_web" / "data.csv", index=False, encoding="utf-8")

# Формирование evaluation.csv (question, answer, doc_ids)
eval_df = pd.DataFrame(
    {
        "question": list(questions),
        "answer": list(answers),
        "doc_ids": list(range(len(contexts))),
    }
)
eval_df.to_csv(DATASETS_DIR / "falcon_refined_web" / "evaluation.csv", index=False, encoding="utf-8")
