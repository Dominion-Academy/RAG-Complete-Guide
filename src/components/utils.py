from src.components.knowledge_storage import Document


def repair_json_string(text: str) -> str:
    new_text = text[:]
    if new_text.startswith("```json"):
        text = text[len("```json") :]
    if new_text.endswith("```"):
        text = text[: -len("```")]
    new_text = text.strip()
    return new_text


def deduplicate_documents(*documents_results: list[Document]) -> list[Document]:
    seen_doc_ids = set()
    final_documents = []
    for documents in documents_results:
        for document in documents:
            if document.id not in seen_doc_ids:
                seen_doc_ids.add(document.id)
                final_documents.append(document)
    return final_documents
