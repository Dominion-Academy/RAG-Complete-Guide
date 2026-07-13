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
        "You are a query reformulation assistant. "
        "Given a specific question, produce a broader, "
        "more abstract step‑back question that captures the core topic and underlying principles.  "
        "Output ONLY the reformulated question – no explanations, no extra text."
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
    SYSTEM_PROMPT = Template("Please write a short passage to answer the query")
    USER_PROMPT = Template("Original query: $query")

    def __init__(self, llm: SyncOpenAILikeLLM):
        self._llm = llm

    def generate(self, query: str) -> str:
        system_prompt = self.SYSTEM_PROMPT.substitute()
        user_prompt = self.USER_PROMPT.substitute({"query": query})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]

        answer = self._llm.generate_answer(messages)
        return answer.content
