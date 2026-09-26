"""Shared Hunyuan clients (CLAUDE.md §8 — shared file, flag before editing).

Chat goes through langchain_community. Embeddings do not: langchain_community
0.3.2 ships `chat_models/hunyuan.py` and nothing else Hunyuan-shaped, so the
embedding side wraps `tencentcloud-sdk-python` directly.
"""

from functools import lru_cache

from langchain_community.chat_models import ChatHunyuan
from langchain_core.embeddings import Embeddings
from tencentcloud.common import credential
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

from src.config import get_settings

# document_chunks.embedding is VECTOR(1024), so any embedding model used here must
# produce 1024 dimensions or the rows will not fit the column. Two that do:
# Hunyuan ("向量维度为1024维", fixed by the vendor) and Ollama's mxbai-embed-large.
# Changing this means an ALTER on the column and a full re-embed.
EMBEDDING_DIM = 1024

# "输入文本。总长度不超过 1024 个 Token，超过则会截断最后面的内容."
# Chunks longer than this are TRUNCATED SERVER-SIDE, silently. Chunk below it.
EMBEDDING_MAX_INPUT_TOKENS = 1024


@lru_cache(maxsize=1)
def get_chat_model():
    s = get_settings()
    if s.llm_backend == "fake":
        from src.llm.fake import FakeChatModel, warn_fake_backend

        warn_fake_backend()
        return FakeChatModel()
    if s.llm_backend == "ollama":
        from langchain_community.chat_models import ChatOllama

        # temperature 0: the Router picks a label and the Verifier returns a verdict.
        # Neither is a creative task, and determinism makes the demo reproducible.
        return ChatOllama(
            model=s.ollama_model,
            base_url=s.ollama_base_url,
            temperature=0,
            timeout=s.llm_timeout_seconds,
        )
    return ChatHunyuan(
        hunyuan_app_id=s.hunyuan_app_id,
        hunyuan_secret_id=s.hunyuan_secret_id,
        hunyuan_secret_key=s.hunyuan_secret_key,
        model=s.hunyuan_chat_model,
        streaming=False,
    )


def chat_with_logprobs(prompt: str) -> tuple[str, list[dict]] | None:
    """One chat call that also returns the per-token log-probabilities, or None.

    Only Ollama is wired: langchain's ChatOllama has no logprob passthrough, so this
    posts to the same server's OpenAI-compatible endpoint instead. Returns None for
    any other backend, and on any transport or shape error, so a caller that wants
    measured confidence can fall back to an ordinary call rather than break.

    Same model, same temperature, one round-trip — logprobs ride along with the
    completion, so this costs no extra latency over get_chat_model().invoke().
    """
    s = get_settings()
    if s.llm_backend != "ollama":
        return None

    import httpx

    try:
        response = httpx.post(
            f"{s.ollama_base_url}/v1/chat/completions",
            json={
                "model": s.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 5,
            },
            timeout=s.llm_timeout_seconds,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        tokens = (choice.get("logprobs") or {}).get("content")
    except Exception:
        return None

    return (content, tokens) if tokens else None


class HunyuanEmbeddings(Embeddings):
    """langchain's Embeddings interface over Hunyuan's GetEmbedding.

    GetEmbedding accepts a single `Input` string per call — "当前不支持批量" — so
    `embed_documents` is a loop of round-trips, one per chunk. Ingestion should
    expect network time proportional to chunk count and rate-limit accordingly.
    """

    def __init__(self) -> None:
        s = get_settings()
        cred = credential.Credential(s.hunyuan_secret_id, s.hunyuan_secret_key)
        self._client = hunyuan_client.HunyuanClient(cred, s.hunyuan_region)

    def embed_query(self, text: str) -> list[float]:
        req = models.GetEmbeddingRequest()
        req.Input = text
        resp = self._client.GetEmbedding(req)
        if not resp.Data:
            raise RuntimeError(f"Hunyuan returned no embedding (request {resp.RequestId})")
        vector = list(resp.Data[0].Embedding)
        if len(vector) != EMBEDDING_DIM:
            # Would corrupt every row silently against a VECTOR(1024) column.
            raise RuntimeError(
                f"Hunyuan returned a {len(vector)}-dim embedding, schema expects {EMBEDDING_DIM}"
            )
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # ponytail: serial round-trips, one per text — the API has no batch form.
        # Parallelise here if ingestion time becomes the bottleneck.
        return [self.embed_query(t) for t in texts]


@lru_cache(maxsize=1)
def get_embeddings():
    """Embeddings for the active backend. Both options are 1024-dim (EMBEDDING_DIM).

    Ollama needs the model pulled first: `ollama pull mxbai-embed-large`.
    """
    s = get_settings()
    if s.llm_backend == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model=s.ollama_embedding_model, base_url=s.ollama_base_url)
    return HunyuanEmbeddings()
