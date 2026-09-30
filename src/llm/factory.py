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


class ModelUnavailable(RuntimeError):
    """The language model could not be reached. Raised HERE, at the call, because
    this is the last place the truth survives.

    Measured: a dead Ollama raises `requests.ConnectionError`, langchain catches it
    and re-raises `ValueError("Error raised by inference endpoint: ...")`, and anyio
    then carries that across the threadpool through a task group — which OVERWRITES
    `__context__` with its own ExceptionGroup. By the time the API's error handler
    sees it the chain reads `ValueError -> ExceptionGroup -> ValueError -> ...` and
    the requests error is gone. So the fault cannot be classified downstream from
    what arrives; it has to be named where it happens.

    An exception's TYPE survives all of that, which is the whole point of this class.
    """


_TRANSPORT_MODULES = frozenset({"requests", "httpx", "urllib3", "http", "socket", "ssl"})


def _is_transport_fault(exc: BaseException) -> bool:
    """A network fault reaching the model, rather than a bad answer from it.

    ponytail: matched on the exception's top-level module, because the HTTP client
    behind a chat model changes with the backend and is not worth importing three
    libraries to name precisely. Walks causes and contexts, which ARE intact here —
    it is only the trip through the threadpool that destroys them.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__module__.split(".")[0] in _TRANSPORT_MODULES:
            return True
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class _Guarded:
    """Passes everything through, turning a transport fault into ModelUnavailable.

    One wrapper at the factory rather than a try/except at each of the five call
    sites, which is five places to forget.
    """

    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        """The model underneath, for tests that assert on the backend's own type."""
        return self._inner

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def guarded(*args, **kwargs):
            try:
                return attribute(*args, **kwargs)
            except ModelUnavailable:
                raise
            except Exception as exc:
                if _is_transport_fault(exc):
                    raise ModelUnavailable(str(exc)) from exc
                raise

        return guarded


@lru_cache(maxsize=1)
def get_chat_model():
    s = get_settings()
    if s.llm_backend == "fake":
        from src.llm.fake import FakeChatModel, warn_fake_backend

        warn_fake_backend()
        return FakeChatModel()  # not guarded: it cannot make a network call
    if s.llm_backend == "ollama":
        from langchain_community.chat_models import ChatOllama

        # temperature 0: the Router picks a label and the Verifier returns a verdict.
        # Neither is a creative task, and determinism makes the demo reproducible.
        return _Guarded(ChatOllama(
            model=s.ollama_model,
            base_url=s.ollama_base_url,
            temperature=0,
            timeout=s.llm_timeout_seconds,
            # Keeps the model resident between questions; see the setting's note.
            keep_alive=s.ollama_keep_alive,
        ))
    return _Guarded(ChatHunyuan(
        hunyuan_app_id=s.hunyuan_app_id,
        hunyuan_secret_id=s.hunyuan_secret_id,
        hunyuan_secret_key=s.hunyuan_secret_key,
        model=s.hunyuan_chat_model,
        streaming=False,
    ))


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


class OllamaKeptEmbeddings(Embeddings):
    """Ollama embeddings that stay resident, by calling the API directly.

    langchain_community's OllamaEmbeddings cannot express `keep_alive` — the field
    does not exist and the model forbids extras — so the embedding model fell back to
    Ollama's five-minute default while the chat model was held for thirty. Measured
    with `/api/ps`: qwen2.5:7b expiring at 10:56, mxbai-embed-large at 10:31.

    The Router embeds before anything else runs, so any question after a five-minute
    gap paid to reload it — which is exactly the shape of a demo. One observed Router
    node took 11.7s while the Synthesizer and Verifier beside it took 2.7s each,
    because those two found a chat model that was still warm.

    That wrapper is deprecated anyway, and this is the one setting we actually needed
    from it.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._url = s.ollama_base_url.rstrip("/") + "/api/embeddings"
        self._model = s.ollama_embedding_model
        self._keep_alive = s.ollama_keep_alive
        self._timeout = s.llm_timeout_seconds

    def embed_query(self, text: str) -> list[float]:
        import requests

        response = requests.post(
            self._url,
            json={"model": self._model, "prompt": text, "keep_alive": self._keep_alive},
            timeout=self._timeout,
        )
        response.raise_for_status()
        vector = response.json().get("embedding") or []
        if len(vector) != EMBEDDING_DIM:
            # Would corrupt every row silently against a VECTOR(1024) column.
            raise RuntimeError(
                f"Ollama returned a {len(vector)}-dim embedding, schema expects {EMBEDDING_DIM}"
            )
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # ponytail: serial, like the Hunyuan one — /api/embeddings takes one prompt.
        # Parallelise here if ingestion time becomes the bottleneck.
        return [self.embed_query(t) for t in texts]


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
        # OllamaKeptEmbeddings rather than langchain's wrapper, so OLLAMA_KEEP_ALIVE
        # applies to this model too. "A reload is milliseconds, not seconds" was the
        # reason this did not matter; measured, it is seconds, and it lands on the
        # Router — the first node of every request.
        return _Guarded(OllamaKeptEmbeddings())
    return _Guarded(HunyuanEmbeddings())
