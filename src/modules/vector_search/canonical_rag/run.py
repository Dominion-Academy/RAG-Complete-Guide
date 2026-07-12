from string import Template

from qdrant_client import QdrantClient

from src.components.embeddings import FastEmbedEmbeddingsModel
from src.components.knowledge_storage import QdrantCollectionKnowledgeStorage, SimpleRetriever
from src.components.llm import InputMessage, SyncOpenAILikeLLM
from src.settings import settings


# Настройка параметров
TOP_K = 3
QUERY = "What is the Berry Export Summary 2028 and what is its purpose?"


# Промпты
SYSTEM_PROMPT = Template(
    "You are a useful assistant. Use the language specified in $lang for your entire response. "
    "Answer the user's question briefly using only contextual data."
)
USER_PROMPT = Template("Question: $query\n\nContext (the ONLY source of truth):\n$context")


# 1. Инициализация компонентов
knowledge_storage = QdrantCollectionKnowledgeStorage(
    embeddings_model=FastEmbedEmbeddingsModel(),
    client=QdrantClient(url="http://localhost:6333"),
    collection_name="canonical_rag_documents",
)
retriever = SimpleRetriever(storage=knowledge_storage)
llm = SyncOpenAILikeLLM(
    base_url=settings.llm.BASE_URL,
    api_key=settings.llm.API_KEY,
    model=settings.llm.MODEL,
    common_parameters={"temperature": 0},
)

# 2. Поиск фрагментов документов
external_knowledge = retriever.retrieve(QUERY, top_k=TOP_K)
print(f"\nChunks:\n{'\n'.join([str(ek.metadata) + '\n' + ek.text for ek in external_knowledge])}")

# 3. Подготовка сообщений для LLM
system_prompt = SYSTEM_PROMPT.substitute({"lang": "english"})
user_prompt = USER_PROMPT.substitute({"query": QUERY, "context": "\n\n".join([ek.text for ek in external_knowledge])})
messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]

# 4. Вызов LLM
answer = llm.generate_answer(messages)
print(f"\nAnswer:\n{answer.content}\nInput tokens: {answer.input_tokens}\nOutput tokens: {answer.output_tokens}")
