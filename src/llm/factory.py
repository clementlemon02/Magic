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

# Fixed by the vendor API, not by us: "腾讯混元 Embedding 接口 ... 向量维度为1024维".
# document_chunks.embedding is VECTOR(1024) to match — change one, change both.
EMBEDDING_DIM = 1024

# "输入文本。总长度不超过 1024 个 Token，超过则会截断最后面的内容."
# Chunks longer than this are TRUNCATED SERVER-SIDE, silently. Chunk below it.
EMBEDDING_MAX_INPUT_TOKENS = 1024


@lru_cache(maxsize=1)
def get_chat_model() -> ChatHunyuan:
    s = get_settings()
    return ChatHunyuan(
        hunyuan_app_id=s.hunyuan_app_id,
        hunyuan_secret_id=s.hunyuan_secret_id,
        hunyuan_secret_key=s.hunyuan_secret_key,
        model=s.hunyuan_chat_model,
        streaming=False,
    )


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
def get_embeddings() -> HunyuanEmbeddings:
    return HunyuanEmbeddings()
