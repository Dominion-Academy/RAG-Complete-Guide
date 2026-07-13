from src.components.llm import SyncOpenAILikeLLM
from src.components.pre_retrieval import StepBackPromptingRewriter
from src.settings import settings


# 1. Инициализация компонентов
llm = SyncOpenAILikeLLM(
    base_url=settings.llm.BASE_URL,
    api_key=settings.llm.API_KEY,
    model=settings.llm.MODEL,
    common_parameters={"temperature": 0},
)
query_rewriter = StepBackPromptingRewriter(llm)

# 2. Подготовка запроса пользователя
QUERY = "What is the Berry Export Summary 2028 and what is its purpose?"


# 3. Перефразирование вопроса
abstract_query = query_rewriter.generate(QUERY)
print(abstract_query)
