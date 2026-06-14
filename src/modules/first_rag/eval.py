import csv

import asyncclick as click

from src.modules.first_rag.app import DATASET_DIR, NGramKnowledgeStorage, NGramRetriever


def compute_hit_rate(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    return bool(set(retrieved_ids) & set(relevant_ids))


@click.command()
@click.option("-n", default=3, type=int, help="Character n-gram size (для NGramRetriever)")
@click.option("-k", default=2, type=int, help="Number of articles to retrieve (k)")
async def main(n: int, k: int) -> None:
    data_path = DATASET_DIR / "ru.txt"
    knowledge_store = NGramKnowledgeStorage(n=n).load_documents_from_file(data_path)
    retriever = NGramRetriever(storage=knowledge_store)

    # Questions loading
    questions = []
    csv_path = DATASET_DIR / "evaluation.csv"
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "question" not in reader.fieldnames or "doc_ids" not in reader.fieldnames:
            click.echo("CSV должен содержать колонки: question, answer, doc_ids", err=True)
            return
        for row in reader:
            expected_ids = {int(x.strip()) for x in row["doc_ids"].split(",") if x.strip()}
            questions.append(
                {
                    "question": row["question"],
                    "expected_ids": expected_ids,
                    "answer": row.get("answer", ""),
                }
            )

    # Evaluation
    click.echo("===== EVALUATION =====")
    total_hit = 0.0
    for idx, q in enumerate(questions, 1):
        query = q["question"]
        expected_ids = q["expected_ids"]
        retrieved_ids = [knowledge.doc_id for knowledge in retriever.retrieve(query, top_k=k)]
        hit_rate = bool(set(retrieved_ids) & set(expected_ids))
        total_hit += hit_rate
        click.echo(f"{idx}. {'✓' if hit_rate else '✗'} | expected={expected_ids} got={retrieved_ids}")

    # Finalization
    n_queries = len(questions)
    hit_rate_total = total_hit / n_queries if n_queries else 0

    click.echo("\n===== FINAL RESULTS =====")
    click.echo(f"N queries: {n_queries}")
    click.echo(f"Hit Rate@{k}: {hit_rate_total:.2f}")


if __name__ == "__main__":
    main()
