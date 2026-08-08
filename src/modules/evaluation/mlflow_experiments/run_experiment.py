import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Correctness

from src.modules.evaluation.rag_pipelines.simple import pipeline


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id="5")
DATASET_ID = "d-0c4e5793dfb24d14b91d4a364ba8fe0f"


scorers = [Correctness(model="gateway:/my-model")]


def call_simple_rag(question: str) -> str:
    answer = pipeline.generate(question, top_k=6)
    return answer.content


if __name__ == "__main__":
    dataset = get_dataset(dataset_id=DATASET_ID)
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=call_simple_rag,
        scorers=scorers,
    )
