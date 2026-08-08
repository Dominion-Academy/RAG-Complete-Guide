import os

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase

from src.modules.evaluation.rag_pipelines.simple import pipeline
from src.settings import settings


os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"

MODEL = LocalModel(
    model=settings.llm.MODEL,
    api_key=settings.llm.API_KEY,
    base_url=settings.llm.BASE_URL,
)
RAG_TRIADE_METRICS = [
    AnswerRelevancyMetric(model=MODEL, async_mode=False),
    FaithfulnessMetric(model=MODEL, async_mode=False),
    ContextualRelevancyMetric(model=MODEL, async_mode=False),
]

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    "./dataset/golden_dataset.json", input_key_name="input", expected_output_key_name="expected_output"
)

test_cases = []
for golden in dataset.goldens:
    retrieval_context = [doc.text for doc in pipeline.retrieve(golden.input)]
    answer = pipeline.generate(golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer.content,
        expected_output=golden.expected_output,
        retrieval_context=retrieval_context,
    )
    dataset.add_test_case(test_case)

evaluate(
    test_cases=dataset.test_cases[:5], metrics=RAG_TRIADE_METRICS, async_config=AsyncConfig(max_concurrent=1, run_async=False)
)
