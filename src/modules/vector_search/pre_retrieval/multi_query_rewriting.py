from src.components.llm import SyncOpenAILikeLLM
from src.components.pre_retrieval import MultiQueryRewriter
from src.settings import settings


# 1. Инициализация компонентов
llm = SyncOpenAILikeLLM(
    base_url=settings.llm.BASE_URL,
    api_key=settings.llm.API_KEY,
    model=settings.llm.MODEL,
    common_parameters={"temperature": 0},
)
query_rewriter = MultiQueryRewriter(llm)

# 2. Подготовка запроса пользователя
QUERY = "What is the Berry Export Summary 2028 and what is its purpose?"


# 3. Перефразирование вопроса
alternative_queries = query_rewriter.generate(QUERY)
for query in alternative_queries:
    print(query)
