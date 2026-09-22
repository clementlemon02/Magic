"""The offline backend switch. It must be opt-in, and it must announce itself."""

import pytest

from src.config import get_settings
from src.llm.fake import FakeChatModel, UnknownPrompt
from src.llm.factory import get_chat_model


@pytest.fixture(autouse=True)
def clear_caches():
    get_settings.cache_clear()
    get_chat_model.cache_clear()
    yield
    get_settings.cache_clear()
    get_chat_model.cache_clear()


def test_fake_backend_is_opt_in_and_warns(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "fake")
    with pytest.warns(RuntimeWarning, match="canned"):
        model = get_chat_model()
    assert isinstance(model, FakeChatModel)


def test_default_backend_is_ollama():
    """The only backend that can actually run — no Hunyuan credentials exist.

    Constructed with _env_file=None so this asserts the code default rather than
    whatever a developer happens to have in their local .env.
    """
    from src.config import Settings

    assert Settings(_env_file=None).llm_backend == "ollama"


def test_ollama_backend_is_selectable_without_warning(monkeypatch):
    """Constructing the client does not connect, so this needs no running server."""
    from langchain_community.chat_models import ChatOllama

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    model = get_chat_model()
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen2.5:7b"
    assert model.temperature == 0  # deterministic: routing and judging aren't creative


def test_unrecognised_prompt_raises_rather_than_guessing():
    """A reworded PROMPT_TEMPLATE should break the fake loudly, not silently mis-answer."""
    with pytest.raises(UnknownPrompt):
        FakeChatModel().invoke("some prompt this fake has never seen")
