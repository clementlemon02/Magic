"""Env-backed settings. CLAUDE.md §7: declared in .env.example, never hardcoded.

One module so the three workstreams share defaults instead of each writing their
own os.getenv fallback and drifting from .env.example.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://internal_brain:change-me@localhost:5432/internal_brain"

    # Tencent Cloud IAM credentials, not the OpenAI-compatible key. See .env.example.
    hunyuan_app_id: str = ""
    hunyuan_secret_id: str = ""
    hunyuan_secret_key: str = ""
    hunyuan_chat_model: str = "hunyuan-turbo"
    hunyuan_region: str = ""  # Hunyuan is region-agnostic; kept for SDK signature

    retrieval_top_k: int = 6
    retrieval_max_hops: int = 3
    retrieval_min_score: float = 0.72
    verifier_confidence_threshold: float = 0.6
    permission_conflict_score_margin: float = 0.05

    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
