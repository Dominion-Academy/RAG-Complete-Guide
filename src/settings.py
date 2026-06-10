from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=ROOT_DIR / ".env", extra="ignore")

    BASE_URL: str = Field(description="Эндпоинт для доступа к LLM")
    API_KEY: str = Field(description="Ключ для доступа к LLM")
    MODEL: str = Field(description="Название модели LLM")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=ROOT_DIR / ".env", extra="ignore")

    llm: LLMConfig = LLMConfig()


settings = Settings()
