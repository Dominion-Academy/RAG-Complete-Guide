from string import Template

from src.components.llm import AIAnswer, InputMessage, SyncOpenAILikeLLM
from src.components.sql_query_executor import BaseSQLExecutor
from src.components.sql_schema_extractors import BaseSQLSchemaExtractor
from src.components.utils import repair_sql_string


class SQLRAG:
    GEN_SQL_PROMPT = Template(
        "You need to create an SQL query to answer the question. "
        "Question: $query "
        "Use only the tables and columns provided in the schema. "
        "Do not use any other tables or columns that are not provided in the schema. "
        "Generate only a read-only SELECT query. "
        "$db_schema "
        "ONLY RESPOND WITH SQL."
    )
    SYSTEM_PROMPT = Template("You are a useful assistant. Answer the user's question briefly using only contextual data.")
    USER_PROMPT = Template("Question: $query\n\nContext (the ONLY source of truth):\n$context")

    def __init__(
        self,
        sql_executor: BaseSQLExecutor,
        schema_extractor: BaseSQLSchemaExtractor,
        llm: SyncOpenAILikeLLM,
    ) -> None:
        self._sql_executor = sql_executor
        self._schema_extractor = schema_extractor
        self._llm = llm

    def _retrieve(self, query: str) -> list[dict]:
        db_schema = self._schema_extractor.get_schema_text()
        prompt = self.GEN_SQL_PROMPT.substitute({"query": query, "db_schema": db_schema})
        sql = (self._llm.generate_answer([InputMessage(role="user", content=prompt)])).content
        repaired_sql = repair_sql_string(sql)
        print("SQL:", repaired_sql)
        return self._sql_executor.execute(repaired_sql)

    def generate(self, query: str) -> AIAnswer:
        records = self._retrieve(query)
        context = str(records)
        system_prompt = self.SYSTEM_PROMPT.substitute()
        user_prompt = self.USER_PROMPT.substitute({"query": query, "context": context})
        messages = [InputMessage(role="system", content=system_prompt), InputMessage(role="user", content=user_prompt)]
        return self._llm.generate_answer(messages)
