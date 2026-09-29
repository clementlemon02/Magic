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


def test_default_backend_is_ollama(monkeypatch):
    """The only backend that can actually run — no Hunyuan credentials exist.

    `_env_file=None` ignores the .env FILE, but pydantic-settings still reads the
    environment, so this also clears the variable. Without that it asserted whatever
    the shell happened to export — green locally, red under CI, which runs the suite
    with LLM_BACKEND=fake because there is no Ollama on the runner.
    """
    from src.config import Settings

    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert Settings(_env_file=None).llm_backend == "ollama"


def test_ollama_backend_is_selectable_without_warning(monkeypatch):
    """Constructing the client does not connect, so this needs no running server."""
    from langchain_community.chat_models import ChatOllama

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    model = get_chat_model()
    # `.inner`: the factory wraps every real model so a transport fault is named
    # where the cause chain still exists (src/llm/factory.ModelUnavailable).
    assert isinstance(model.inner, ChatOllama)
    assert model.model == "qwen2.5:7b"
    assert model.temperature == 0  # deterministic: routing and judging aren't creative


def test_unrecognised_prompt_raises_rather_than_guessing():
    """A reworded PROMPT_TEMPLATE should break the fake loudly, not silently mis-answer."""
    with pytest.raises(UnknownPrompt):
        FakeChatModel().invoke("some prompt this fake has never seen")


def test_the_chat_model_stays_resident_between_questions(monkeypatch):
    """Ollama unloads an idle model after 5 minutes, so the question asked after any
    pause in a demo pays the reload — measured at 5.8s on a refusal whose deadline is
    4.0s, which is an escape on the property being demonstrated."""
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "45m")
    get_chat_model.cache_clear()
    try:
        assert get_chat_model().keep_alive == "45m"
    finally:
        get_chat_model.cache_clear()
