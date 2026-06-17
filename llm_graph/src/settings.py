from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

current_file = Path(__file__).resolve()
current_dir = current_file.parent
env_path = current_dir.parent / ".env" 

class Settings(BaseSettings):
    LANGSMITH_API_KEY: str = "str"
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "str"
    LANGSMITH_PROJECT: str = "str"
    OPENROUTER_API_KEY: str = "str"
    PROXY_API_KEY: str = "str"
    DATADOG_METRICS_ENABLED: bool = False
    LSD_PROM_METRICS_ENABLED: bool = False

    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding="utf-8")

settings = Settings()

if __name__ == "__main__":
    print(settings.model_dump())