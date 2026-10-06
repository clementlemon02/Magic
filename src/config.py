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

    # Distilled Router (scripts/train_router_student.py). Below this probability the
    # LLM Router decides instead. Per-model, like the other similarity floors.
    router_student_enabled: bool = True
    router_student_min_confidence: float = 0.8  # lowest with 0 errors in 5-fold CV

    retrieval_top_k: int = 6
    retrieval_max_hops: int = 3
    # No new retrieval hop starts after this long, so a refusal ends within the budget
    # plus one hop. Keep budget + slowest hop under REFUSAL_DEADLINE_SECONDS. PER-HARDWARE:
    # the slowest single hop measured ~1.9s on an M3 Pro with qwen2.5:7b, so 2.0 puts the
    # ceiling at ~3.9s, inside the 4.0s deadline. At 2.5 it was 4.34s and 2.9% of refusals
    # escaped. Swept with evals/refusal_timing.py; 2.5, 2.0 and 1.5 all answered 40/40, so
    # this cost nothing on THIS corpus — every answer here resolves on its first hop. On a
    # corpus that needs a second hop to answer, lowering this starts refusing them instead.
    retrieval_hop_budget_seconds: float = 2.0
    retrieval_min_score: float = 0.55  # per-model AND per-corpus; see .env.example before changing
    # Skip the Synthesizer's model call when one retrieved sentence, unedited,
    # already leads every other candidate on Jaccard word overlap with the query by
    # this much. The Verifier's own fast path (a verbatim substring check, no
    # setting of its own) then usually fires right after, so a hit here tends to
    # skip both model calls. Measured live 30 Sep against the seeded corpus: a
    # genuine single-sentence answer scores 0.364; the closest live miss — a
    # sentence sharing only the question's topic nouns ("payment", "outage") but
    # not what was actually asked ("caused") — scores 0.167; a compound question
    # answered by only half its evidence scores 0.125. 0.25 sits with margin above
    # both misses and below the one hit. Swept with evals/fast_path_sweep.py;
    # re-measure if the stopword list or the corpus changes meaningfully.
    synthesis_fast_path_min_overlap: float = 0.25
    verifier_confidence_threshold: float = 0.6
    permission_conflict_score_margin: float = 0.05

    # Re-ask the source connector whether it still grants access, at query time, for
    # RESTRICTED documents only (src/connectors/__init__.py). Closes the staleness
    # window between a revocation in the source and our next ingest. Off puts us back
    # on the mirror alone, which is where every other mirrored-ACL product sits.
    source_recheck_enabled: bool = True

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
    # An answer older than this is never served from the cache, whatever else says it is
    # current. The brief bounds how stale an answer may be at about an hour; this is the
    # backstop for any change the corpus version (src/cache.py) does not see.
    query_cache_ttl_seconds: int = 3600

    # How often the server re-reads every source and reconciles documents, chunks and
    # grants (src/ingestion/sync.py). This IS the freshness bound: an edit at a source
    # reaches answers within one interval plus the time a sync takes. 0 turns the schedule
    # off; `POST /admin/sync` and `python -m scripts.sync_sources` still work.
    sync_interval_minutes: int = 10
    # Apply the additive, idempotent schema changes at startup, so a database created
    # before a column existed keeps working. Off in tests, which build their own schema.
    migrate_on_startup: bool = True
    # The mock sources' authored state (scripts/mock_source.py); relative to the repo root.
    mock_sources_path: str = ".mock_sources.json"

    # Cosine floor for grouping unanswered questions into one knowledge gap.
    # Per-model, like retrieval_min_score.
    knowledge_gap_similarity: float = 0.75

    # Load the model at startup instead of on the first question. A cold Ollama took
    # 12.54s on the first request and 1.8s afterwards, and at a demo the first question
    # is the one someone else asks. Off in tests (conftest.py) — every create_app would
    # otherwise reach for a model.
    warm_on_startup: bool = True
    # How long Ollama keeps the model resident between requests. Its own default is
    # 5 minutes, so any pause in a demo unloads it and the next request pays the
    # reload — measured at 5.8s on a refusal whose deadline is 4.0s, which turns the
    # constant-time guarantee into an escape for exactly the question someone asks
    # after a conversation. Warming at startup does not help a model that has since
    # been unloaded.
    ollama_keep_alive: str = "30m"

    # Signing key for the bearer token (src/api/auth.py). The default is a DEMO key
    # and is in the repository on purpose, so a fresh clone runs; set AUTH_SECRET to
    # anything else before this is reachable by someone you did not invite, because
    # whoever holds it can mint a token for any user id.
    auth_secret: str = "demo-only-change-me"
    auth_token_ttl_minutes: int = 720  # a working day, then sign in again
    # Failed sign-ins allowed per email and per client before a lockout window.
    # /auth/login had no limit at all, so scrypt's cost was the only thing between a
    # caller and unlimited password guessing.
    login_max_attempts: int = 5
    login_lockout_seconds: float = 300.0
    # /query, per CALLER. Every request counts, answers and refusals alike — see the
    # note on SlidingWindow for why they must cost the same. Generous for a person
    # (the model itself takes ~5s a question); tight enough that one caller cannot
    # park a held refusal per request and queue the model against everybody else.
    query_max_per_window: int = 20
    query_window_seconds: float = 60.0

    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
