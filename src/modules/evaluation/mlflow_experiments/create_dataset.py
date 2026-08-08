import csv

import mlflow
from mlflow.genai.datasets import create_dataset

from src.settings import DATASETS_DIR


mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_ID = "5"  # Укажите свой
DATASET_PATH = DATASETS_DIR / "ai_helpdesk" / "evaluation.csv"

records = []
with open(DATASET_PATH, encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        new_record = {
            "inputs": {"question": row["question"]},
            "expectations": {"expected_response": row["answer"], "relevant_heading": row["headings"]},
        }
        records.append(new_record)


dataset = create_dataset(
    name="golden_dataset",
    experiment_id=[EXPERIMENT_ID],
)
dataset.merge_records(records)
