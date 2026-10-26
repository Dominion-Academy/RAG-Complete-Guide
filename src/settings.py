from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=ROOT_DIR / ".env", extra="ignore")

    BASE_URL: str = Field(description="Эндпоинт для доступа к LLM")
    API_KEY: str = Field(description="Ключ для доступа к LLM")
    MODEL: str = Field(description="Название модели LLM")


class PostgresConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=ROOT_DIR / ".env", extra="ignore")

    HOST: str = Field(default="localhost", description="Хост PostgreSQL")
    PORT: int = Field(default=5433, description="Порт PostgreSQL (см. docker-compose.yaml)")
    USER: str = Field(default="postgres", description="Пользователь PostgreSQL")
    PASSWORD: str = Field(default="password", description="Пароль PostgreSQL")
    DB: str = Field(default="postgres", description="Название базы данных PostgreSQL")

    def connect_kwargs(self) -> dict:
        return {"host": self.HOST, "port": self.PORT, "user": self.USER, "password": self.PASSWORD, "dbname": self.DB}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=ROOT_DIR / ".env", extra="ignore")

    llm: LLMConfig = LLMConfig()
    postgres: PostgresConfig = PostgresConfig()

    HF_TOKEN: str = Field(default="...", description="Токен доступа Hugging Face")


settings = Settings()

DATASETS_DIR = Path(__file__).parent / "datasets"
