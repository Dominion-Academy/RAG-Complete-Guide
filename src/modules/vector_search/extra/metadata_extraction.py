import json
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from src.components.llm import InputMessage, SyncOpenAILikeLLM
from src.components.utils import repair_json_string
from src.settings import settings


EXTRACT_PROMPT = """
You are a filter extraction module in a search system.  
Your task is to analyze the user’s query and extract **only explicitly mentioned filters** from the metadata fields described below.

Rules:
1. Extract values only from the query text. Do not infer or add filters that are not present.
2. If a filter is not mentioned, do not include it in the output.
3. If no filters are found at all, return an empty JSON object: `{}`.


## Metadata Field Description

### Field `department`
- Type: string or array of strings.
- Allowed values (closed list, DO NOT invent new ones): `["research", "engineering", "marketing"]`
- Synonyms and abbreviations may appear in queries (e.g., “R&D” → `research`, “eng” → `engineering`, “sales” is not allowed). Map them to the allowed values where possible.

### Field `rating`
- Type: number from 0 to 5.
- Return keys `min_rating` and/or `max_rating` based on the expressed condition.

## Example

Query:  
"Мне нужны хорошие статьи из R&D"

Response:  
```json
{
  "department": "research",
  "min_rating": 4
}
"""


class LLMMetadataExtractor:
    def __init__(self, llm: SyncOpenAILikeLLM):
        self._llm = llm

    def _extract_raw_filters(self, query: str) -> dict[str, Any]:
        messages = [InputMessage(role="system", content=EXTRACT_PROMPT), InputMessage(role="user", content=query)]
        answer = self._llm.generate_answer(messages)
        return json.loads(repair_json_string(answer.content))

    def extract(self, query: str) -> Filter:
        raw_filters = self._extract_raw_filters(query)
        print(raw_filters)

        field_conditions = []
        for key, value in raw_filters.items():
            if key == "department" and isinstance(value, str):
                field_conditions.append(
                    FieldCondition(
                        key="department",
                        match=MatchValue(value=value),
                    )
                )
            elif key == "department" and isinstance(value, list):
                field_conditions.append(
                    FieldCondition(
                        key="department",
                        match=MatchAny(any=value),
                    )
                )
            elif key == "min_rating":
                field_conditions.append(
                    FieldCondition(
                        key="rating",
                        range=Range(
                            gt=None,
                            gte=value,
                            lt=None,
                            lte=None,
                        ),
                    )
                )
            elif key == "max_rating":
                field_conditions.append(
                    FieldCondition(
                        key="rating",
                        range=Range(
                            gt=None,
                            gte=None,
                            lt=None,
                            lte=value,
                        ),
                    )
                )
        return Filter(must=field_conditions)


llm = SyncOpenAILikeLLM(
    base_url=settings.llm.BASE_URL,
    api_key=settings.llm.API_KEY,
    model=settings.llm.MODEL,
    common_parameters={"temperature": 0},
)
llm_extractor = LLMMetadataExtractor(llm=llm)

queries = [
    "Мне нужны самые плохие статьи",
    "Лучшие статьи по маркитингу и погромированию",
    "Направление моих исследований это квантовая физика, что вы можете порекомендовать?",
]
for query in queries:
    print(query)
    filters = llm_extractor.extract(query=query)
    print(filters)
    print()
