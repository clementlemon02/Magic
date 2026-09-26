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

    # "hunyuan" (default), "ollama" or "fake". The hackathon provides no Hunyuan
    # credentials, so ollama is the working local default until a hosted provider
    # is chosen; fake answers from canned strings and is for tests only.
    llm_backend: str = "ollama"

    # qwen2.5 rather than qwen3: qwen3 emits <think> blocks by default, which is
    # noise the Router's one-word answer and the Verifier's JSON both have to survive.
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    # 1024-dim, matching document_chunks.embedding VECTOR(1024). Needs `ollama pull`.
    ollama_embedding_model: str = "mxbai-embed-large"
    hunyuan_region: str = ""  # Hunyuan is region-agnostic; kept for SDK signature

    retrieval_top_k: int = 6
    retrieval_max_hops: int = 3
    retrieval_min_score: float = 0.55  # per-model; see .env.example
    verifier_confidence_threshold: float = 0.6
    permission_conflict_score_margin: float = 0.05

    # Nothing may hang forever during a live demo. Both defaults are generous enough
    # for a 7B on a laptop and short enough that a wedged dependency surfaces on stage
    # as a clear error rather than a spinner.
    llm_timeout_seconds: int = 30
    db_connect_timeout_seconds: int = 5

    # Constant-time refusal (docs/design/constant-time-refusal.md). Every refusal is
    # held until this long after the request arrived, so its timing can't reveal
    # whether it was withheld or simply unanswerable. PER-HARDWARE: 4.0s measured on
    # an M3 Pro with qwen2.5:7b; re-measure with evals/refusal_timing.py elsewhere.
    refusal_padding_enabled: bool = True
    refusal_deadline_seconds: float = 4.0

    # Semantic answer cache. The similarity floor is per-model, like
    # retrieval_min_score — re-measure it if the embedding model changes.
    query_cache_enabled: bool = True
    query_cache_similarity: float = 0.93

    # Cosine floor for grouping unanswered questions into one knowledge gap.
    # Per-model, like retrieval_min_score.
    knowledge_gap_similarity: float = 0.75

    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
