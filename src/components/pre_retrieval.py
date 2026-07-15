import json
from string import Template

from src.components.llm import InputMessage, SyncOpenAILikeLLM
from src.components.utils import repair_json_string


class MultiQueryRewriter:
    SYSTEM_PROMPT = Template(
        "Given the original user query, generate 3-5 different rephrasings of it. "
        "Each rephrasing should be a standalone query that captures the same core "
        "intent but uses different wording, synonyms, or focuses on different aspects.\n"
        "Output format: Return a **valid JSON object only**. "
        "Do not add any text, explanations, markdown, or code fences before or after the JSON. "
        "The JSON must have this exact structure: \n"
        "["
        '"First rephrased query",'
        '"Second rephrased query",'
        '"Third rephrased query"'
        "]"
    )
    USER_PROMPT = Template("Original query: $query")

    def __init__(self, llm: SyncOpenAILikeLLM):
        self._llm = llm

    def generate(self, query: str) -> list[str]:
        system_prompt = self.SYSTEM_PROMPT.substitute()
        user_prompt = self.USER_PROMPT.substitute({"query": query})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]

        answer = self._llm.generate_answer(messages)
        return json.loads(repair_json_string(answer.content))


class StepBackPromptingRewriter:
    SYSTEM_PROMPT = Template(
        "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, "
        "which is easier to answer. The step-back question should be broader and cover the underlying principles."
    )
    USER_PROMPT = Template("Original query: $query")

    def __init__(self, llm: SyncOpenAILikeLLM):
        self._llm = llm

    def generate(self, query: str) -> str:
        system_prompt = self.SYSTEM_PROMPT.substitute()
        user_prompt = self.USER_PROMPT.substitute({"query": query})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]

        answer = self._llm.generate_answer(messages)
        return answer.content


class HyDERewriter:
    SYSTEM_PROMPT = Template(
        "Write a detailed, factual passage that answers the given query. "
        "The passage should contain specific entities, numbers, and dates. Do not say: I don't know"
    )
    USER_PROMPT = Template("Original query: $query")

    def __init__(self, llm: SyncOpenAILikeLLM):
        self._llm = llm

    def generate(self, query: str) -> str:
        system_prompt = self.SYSTEM_PROMPT.substitute()
        user_prompt = self.USER_PROMPT.substitute({"query": query})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]

        answer = self._llm.generate_answer(messages)
        return answer.content
