from openai import AsyncOpenAI
from ragas.llms import llm_factory

from src.settings import settings


client = AsyncOpenAI(api_key=settings.llm.API_KEY, base_url=settings.llm.BASE_URL)
ragas_llm = llm_factory("gpt-oss:120b-cloud", provider="openai", client=client)
