import asyncio
from datetime import datetime

from ragas import Dataset, experiment

from src.modules.evaluation.rag_pipelines.simple import pipeline


dataset = Dataset.load(name="first_eval_dataset", backend="local/csv", root_dir=".")


@experiment()
async def my_experiment(row, top_k: int = 5):
    sync_func = lambda question: pipeline.generate(question, top_k=top_k)
    response = await asyncio.to_thread(sync_func, row["question"])

    return {
        **row,
        "response": response.content,
        "top_k": top_k,
    }


async def main():
    for top_k in (1, 3):
        exp_name = f"{datetime.now().strftime('%Y_%M_%d_%H')}_simple_rag_top_k_{top_k}"
        results = await my_experiment.arun(dataset, top_k=top_k, name=exp_name)
        print(results)


if __name__ == "__main__":
    asyncio.run(main())
