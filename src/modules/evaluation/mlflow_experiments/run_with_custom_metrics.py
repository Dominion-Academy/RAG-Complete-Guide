import os
import time
from typing import Any, Literal

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.datasets import get_dataset
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer

from src.modules.evaluation.rag_pipelines.simple import pipeline


os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id="5")
DATASET_ID = "d-0c4e5793dfb24d14b91d4a364ba8fe0f"


greeting_check_judge = make_judge(
    name="greeting check",
    instructions=(
        "Evaluate if the agent response included a greeting at the beginning of the response.\n\nAgent's response: {{ outputs }}"
    ),
    feedback_value_type=Literal["greeting_included", "greeting_not_included"],
    model="gateway:/my-model",
)


@scorer(name="relevant heading usage")
def used_relevant_heading(inputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    documents = pipeline.retrieve(inputs["question"], top_k=3)
    retrieved_headings = list({doc.metadata["heading"] for doc in documents})
    if expectations["relevant_heading"] in retrieved_headings:
        return Feedback(value="yes", rationale="Retrieved headings included relevant heading.")
    return Feedback(value="no", rationale=f"No relevant heading in retrieved: {retrieved_headings}")


scorers = [greeting_check_judge, used_relevant_heading]


def call_simple_rag(question: str) -> str:
    answer = pipeline.generate(question, top_k=3)
    time.sleep(1)
    return answer.content


if __name__ == "__main__":
    dataset = get_dataset(dataset_id=DATASET_ID)
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=call_simple_rag,
        scorers=scorers,
    )
