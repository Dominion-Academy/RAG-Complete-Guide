import asyncio

from ragas.metrics import discrete_metric

from src.modules.evaluation.rag_pipelines.simple import pipeline


@discrete_metric(name="greeting_checking", allowed_values=["pass", "fail"])
def greeting_checking_metric(response: str) -> str:
    return "pass" if "Hello" in response else "fail"


async def main():
    question = "В каком парке родилась идея талисмана Debug?"

    answer = pipeline.generate(question, top_k=5)

    result = greeting_checking_metric.score(response=answer.content)
    print(f"Greeting Check: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())
