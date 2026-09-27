from string import Template

from src.components.graph_query_executor import BaseCypherExecutor
from src.components.graph_schema_extractors import BaseSchemaExtractor
from src.components.llm import AIAnswer, InputMessage, SyncOpenAILikeLLM
from src.components.utils import repair_cypher_string


class GraphRAG:
    GEN_CYPHER_PROMPT = Template(
        "You need to create Cypher query to answer the question. "
        "Question: $query "
        "Use only the provided relationship types and properties in the schema. "
        "Do not use any other relationship types or properties that are not provided in the schema. "
        "$graph_schema "
        "ONLY RESPOND WITH CYPHER."
    )
    SYSTEM_PROMPT = Template("You are a useful assistant. Answer the user's question briefly using only contextual data.")
    USER_PROMPT = Template("Question: $query\n\nContext (the ONLY source of truth):\n$context")

    def __init__(
        self,
        cypher_executor: BaseCypherExecutor,
        schema_extractor: BaseSchemaExtractor,
        llm: SyncOpenAILikeLLM,
    ) -> None:
        self._cypher_executor = cypher_executor
        self._schema_extractor = schema_extractor
        self._llm = llm

    def _retrieve(self, query: str) -> list[dict]:
        graph_schema = self._schema_extractor.get_schema_text()
        prompt = self.GEN_CYPHER_PROMPT.substitute({"query": query, "graph_schema": graph_schema})
        cypher = (self._llm.generate_answer([InputMessage(role="user", content=prompt)])).content
        repaired_cypher = repair_cypher_string(cypher)
        print("CYPHER:", repaired_cypher)
        return self._cypher_executor.execute(repaired_cypher)

    def generate(self, query: str) -> AIAnswer:
        records = self._retrieve(query)
        context = str(records)
        system_prompt = self.SYSTEM_PROMPT.substitute()
        user_prompt = self.USER_PROMPT.substitute({"query": query, "context": context})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]
        return self._llm.generate_answer(messages)
