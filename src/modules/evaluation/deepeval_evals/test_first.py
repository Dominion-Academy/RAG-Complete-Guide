from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.modules.evaluation.rag_pipelines.simple import pipeline
from src.settings import settings


model = LocalModel(
    model=settings.llm.MODEL,
    api_key=settings.llm.API_KEY,
    base_url=settings.llm.BASE_URL,
)


def test_correctness():
    question = "В каком парке родилась идея талисмана Debug?"
    reference = "В национальном парке Хванге (Hwange National Park), Зимбабве"
    answer = pipeline.generate(question)

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=model,
    )
    test_case = LLMTestCase(input=question, actual_output=answer.content, expected_output=reference)
    assert_test(test_case, [correctness_metric])
