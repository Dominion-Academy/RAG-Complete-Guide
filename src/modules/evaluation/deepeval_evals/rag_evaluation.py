from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase

from src.modules.evaluation.rag_pipelines.simple import pipeline
from src.settings import settings


MODEL = LocalModel(
    model=settings.llm.MODEL,
    api_key=settings.llm.API_KEY,
    base_url=settings.llm.BASE_URL,
)
RAG_METRICS = [
    AnswerRelevancyMetric(model=MODEL),
    FaithfulnessMetric(model=MODEL),
    ContextualRelevancyMetric(model=MODEL),
    ContextualRecallMetric(model=MODEL),
    ContextualPrecisionMetric(model=MODEL),
]
RECORDS = [
    {
        "question": "В каком парке родилась идея талисмана Debug?",
        "reference": "В национальном парке Хванге (Hwange National Park), Зимбабве",
    },
    {
        "question": "Можно ли закрыть тикет #0000?",
        "reference": "Нет, ни при каких обстоятельствах и ни для одной роли, включая Администратора",
    },
]

test_cases = []
for record in RECORDS:
    retrieval_context = [doc.text for doc in pipeline.retrieve(record["question"])]
    answer = pipeline.generate(record["question"])
    test_case = LLMTestCase(
        input=record["question"],
        actual_output=answer.content,
        expected_output=record["reference"],
        retrieval_context=retrieval_context,
    )
    test_cases.append(test_case)

evaluate(test_cases=test_cases, metrics=RAG_METRICS, async_config=AsyncConfig(max_concurrent=1, run_async=False))
