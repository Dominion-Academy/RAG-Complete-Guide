import asyncio

from ragas.metrics.collections import ContextPrecision

from src.modules.evaluation.rag_pipelines.simple import pipeline
from src.modules.evaluation.ragas_evals.common import ragas_llm


async def main():
    question = "В каком парке родилась идея талисмана Debug?"
    reference = "В национальном парке Хванге (Hwange National Park), Зимбабве"

    documents = pipeline.retrieve(question, top_k=5)

    scorer = ContextPrecision(llm=ragas_llm)
    result = await scorer.ascore(user_input=question, reference=reference, retrieved_contexts=[doc.text for doc in documents])
    print(f"Context Precision Score: {result.value:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
