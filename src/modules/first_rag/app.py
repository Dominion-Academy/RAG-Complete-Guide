from collections import defaultdict
from dataclasses import dataclass
from string import Template
from typing import TYPE_CHECKING, Self

import asyncclick as click

from src.components.llm import InputMessage, OpenAILikeLLM
from src.settings import DATASETS_DIR, settings


if TYPE_CHECKING:
    from pathlib import Path

DATASET_DIR = DATASETS_DIR / "qi_lir_news"


class NGramKnowledgeStorage:
    def __init__(self, n: int) -> None:
        self.n = n
        self.documents = {}  # doc_id: text
        self.inverted_index = defaultdict(set)  # ngram: set(doc_ids)

    def load_documents_from_file(self, data_path: Path) -> Self:
        with open(data_path, encoding="utf-8") as f:
            documents = f.read().split("\n\n")

        for index, text in enumerate(documents):
            doc_id = index + 1
            self.documents[doc_id] = text
            ngrams = self.extract_ngrams(text)
            for ngram in ngrams:
                self.inverted_index[ngram].add(doc_id)

        return self

    def extract_ngrams(self, text: str) -> set[str]:
        ngrams = set()
        for i in range(len(text) - self.n + 1):
            ngrams.add(text[i : i + self.n])
        return ngrams

    def get_document_text(self, doc_id: int) -> str:
        return self.documents[doc_id]

    def check_ngram(self, ngram: str) -> bool:
        return ngram in self.inverted_index

    def get_doc_ids_by_ngram(self, ngram: str) -> set[int]:
        return self.inverted_index[ngram]


@dataclass
class RetrieveResult:
    doc_id: int
    score: int
    text: str


class NGramRetriever:
    def __init__(self, storage: NGramKnowledgeStorage) -> None:
        self.storage = storage

    def retrieve(self, query: str, top_k: int) -> list[RetrieveResult]:
        query_ngrams = self.storage.extract_ngrams(query)

        candidate_docs = defaultdict(int)  # doc_id: count
        for ngram in query_ngrams:
            if self.storage.check_ngram(ngram):
                for doc_id in self.storage.get_doc_ids_by_ngram(ngram):
                    candidate_docs[doc_id] += 1

        sorted_docs = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]

        return [RetrieveResult(doc_id, count, self.storage.get_document_text(doc_id)) for doc_id, count in top_docs]


SYSTEM_PROMPT = Template("You are a useful assistant. Use the language specified in $lang for your entire response.")
USER_PROMPT = Template("Question: $query\n\nContext (the ONLY source of truth):\n$context")


# ------------------------ CLI ------------------------
@click.command()
@click.option("-n", default=3, type=int, help="Character n-gram size (used for retrieval method)")
@click.option("-k", default=3, type=int, help="Number of articles to retrieve")
@click.option("--query", prompt="Your question", help="Question for the RAG system")
async def main(n: int, k: int, query: str) -> None:
    data_path = DATASET_DIR / "ru.txt"

    # Components initialization
    knowledge_store = NGramKnowledgeStorage(n=n).load_documents_from_file(data_path)
    retriever = NGramRetriever(storage=knowledge_store)
    llm = OpenAILikeLLM(
        base_url=settings.llm.BASE_URL,
        api_key=settings.llm.API_KEY,
        model=settings.llm.MODEL,
        common_parameters={"temperature": 0},
    )

    # Retrieve external knowledge
    external_knowledge = retriever.retrieve(query, top_k=k)
    click.echo("==== EXTERNAL KNOWLEDGE ====")
    for knowledge in external_knowledge:
        click.echo(f"doc_id: {knowledge.doc_id}, score: {knowledge.score}\n{knowledge.text}\n")

    # Build prompt
    system_prompt = SYSTEM_PROMPT.substitute({"lang": "russian"})
    user_prompt = USER_PROMPT.substitute({"query": query, "context": "\n\n".join([ek.text for ek in external_knowledge])})
    messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]
    click.echo(f"==== SYSTEM PROMPT ====\n{system_prompt}\n")
    click.echo(f"==== USER PROMPT ====\n{user_prompt}\n")

    # Call LLM
    answer = await llm.generate_answer(messages)
    click.echo(f"==== ANSWER ====\n{answer.content}\nInput tokens: {answer.input_tokens}\nOutput tokens: {answer.output_tokens}")


if __name__ == "__main__":
    main()
