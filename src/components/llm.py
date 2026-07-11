import logging
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class OpenAILikeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class InputMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(description="Роль")
    content: str = Field(description="Текст сообщения")


class AIAnswer(BaseModel):
    content: str = Field(description="Текст ответа")
    input_tokens: int = Field(description="Количество входных токенов")
    output_tokens: int = Field(description="Количество выходных токенов")

    @property
    def message(self) -> InputMessage:
        return InputMessage(role="assistant", content=self.content)


class OpenAILikeLLM:
    def __init__(self, base_url: str, api_key: str, model: str, common_parameters: dict[str, Any]) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.common_parameters = common_parameters

    async def generate_answer(self, messages: list[InputMessage], **kwargs: Any) -> AIAnswer:
        response = await self.client.chat.completions.create(
            model=self.model, messages=[msg.model_dump() for msg in messages], **self.common_parameters, **kwargs
        )
        if response.choices[0].message.content is None:
            raise OpenAILikeError(message="Провайдер не отправил текст в ответе")
        return AIAnswer(
            content=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


class SyncOpenAILikeLLM:
    def __init__(self, base_url: str, api_key: str, model: str, common_parameters: dict[str, Any]) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.common_parameters = common_parameters

    def generate_answer(self, messages: list[InputMessage], **kwargs: Any) -> AIAnswer:
        response = self.client.chat.completions.create(
            model=self.model, messages=[msg.model_dump() for msg in messages], **self.common_parameters, **kwargs
        )
        if response.choices[0].message.content is None:
            raise OpenAILikeError(message="Провайдер не отправил текст в ответе")
        return AIAnswer(
            content=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
